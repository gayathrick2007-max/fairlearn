"""
FairLoan pipeline: catch bias -> fix it -> prove the fix worked.

Everything the dashboard shows is computed here, so the notebook, the training
script and the Streamlit app all agree.

Conventions
-----------
* target y = 1  -> applicant repaid (a "good" loan, should be APPROVED)
* group  a = 1  -> female applicant, a = 0 -> male applicant
* every "gap" is  male - female  (positive = women are disadvantaged)
* Metrics are always measured on a CLEAN test set (true labels). Only the
  TRAINING labels are corrupted, so we know for certain the model is learning
  an injected unfairness and not a real signal.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).parent / "data" / "india_credit.csv"
SENSITIVE_SOURCE = "Gender"  # the gender column in the CSV (Male / Female)
DEFAULT_BIAS = 0.30  # share of truly-good women whose TRAINING label is flipped to "default"
SHOWCASE_SEED = 8  # a typical split (close to the 20-split average), used only to illustrate one person
MODELS = ["Baseline", "Fixed (Fairlearn)"]
EG_NAME = "Fairlearn in-training (no gender at decision time)"
SUBGROUP_COLS = ["Property_Area", "Married", "Education", "Self_Employed"]
DIAL_CUTS = [round(x, 2) for x in np.arange(0.20, 0.661, 0.02)]


# --------------------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Loan_ID"], errors="ignore")
    # Never guess the protected attribute: applicants with no recorded gender are dropped.
    df = df.dropna(subset=[SENSITIVE_SOURCE]).reset_index(drop=True)
    # Other blanks: median for numbers, most common value for text.
    for c in df.columns:
        if df[c].isna().any():
            fill = df[c].median() if pd.api.types.is_numeric_dtype(df[c]) else df[c].mode()[0]
            df[c] = df[c].fillna(fill)
    df["female"] = (df[SENSITIVE_SOURCE].astype(str).str.lower() == "female").astype(int)
    return df


def split_xy(df: pd.DataFrame):
    y = df["credit_risk"].to_numpy()
    a = df["female"].to_numpy()
    X = df.drop(columns=["credit_risk", SENSITIVE_SOURCE])  # keeps 'female' as a feature
    return X, y, a


def feature_lists(X: pd.DataFrame):
    cat = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in ("str", "string")]
    num = [c for c in X.columns if c not in cat]
    return cat, num


def make_preprocessor(X: pd.DataFrame, use_sex: bool = True) -> ColumnTransformer:
    cols = [c for c in X.columns if use_sex or c != "female"]
    cat, num = feature_lists(X[cols])
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", StandardScaler(), num),
        ]
    )


def split_data(seed: int):
    X, y, a = split_xy(load_data())
    Xtr, Xte, ytr, yte, atr, ate = train_test_split(
        X, y, a, test_size=0.3, random_state=seed, stratify=y * 2 + a
    )
    return Xtr, Xte, ytr, yte, atr, ate


# ------------------------------------------------------------------ bias injection
def inject_bias(y: np.ndarray, a: np.ndarray, p: float, seed: int) -> np.ndarray:
    """Historical unfairness: a fraction p of women who DID repay were recorded as defaults.

    Nothing about their ability to repay changed - only the label. This is the
    'old data with unfair decisions in it' from the problem statement.
    """
    rng = np.random.default_rng(seed + 12345)
    y_b = y.copy()
    flip = (a == 1) & (y == 1) & (rng.random(len(y)) < p)
    y_b[flip] = 0
    return y_b


# --------------------------------------------------------------------------- models
def new_lr() -> LogisticRegression:
    return LogisticRegression(max_iter=3000, C=0.5)


class FairLoanModels:
    """Baseline + mitigated model trained on the same (biased) labels."""

    def __init__(self, seed: int = SHOWCASE_SEED, bias: float = DEFAULT_BIAS):
        self.seed, self.bias = seed, bias

    # -- training -------------------------------------------------------------
    def fit(self):
        Xtr, Xte, ytr, yte, atr, ate = split_data(self.seed)
        self.Xtr_raw, self.Xte_raw = Xtr, Xte
        self.ytr_true, self.yte, self.atr, self.ate = ytr, yte, atr, ate
        self.ytr_biased = inject_bias(ytr, atr, self.bias, self.seed)

        self.pre = make_preprocessor(Xtr).fit(Xtr)
        self.Xtr, self.Xte = self.pre.transform(Xtr), self.pre.transform(Xte)

        # 1. Baseline: an ordinary model trained on the biased history
        self.baseline = new_lr().fit(self.Xtr, self.ytr_biased)

        # 2. Fixed: Fairlearn ThresholdOptimizer, demographic parity, on top of the SAME model
        self.fixed = ThresholdOptimizer(
            estimator=self.baseline,
            constraints="demographic_parity",
            objective="accuracy_score",
            prefit=True,
            predict_method="predict_proba",
        ).fit(self.Xtr, self.ytr_biased, sensitive_features=self.atr)

        # 3. Reference: what the model would do if history had been fair (never available in real life)
        self.reference = new_lr().fit(self.Xtr, self.ytr_true)

        # 4. "Just delete the gender column" - the naive fix
        self.pre_unaware = make_preprocessor(Xtr, use_sex=False).fit(Xtr.drop(columns="female"))
        self.unaware = new_lr().fit(
            self.pre_unaware.transform(Xtr.drop(columns="female")), self.ytr_biased
        )

        # 5. Fairness built in DURING training (ExponentiatedGradient). Gender is used only to train;
        #    the finished model never sees it, so it needs no gender at decision time.
        self.eg = ExponentiatedGradient(
            new_lr(), DemographicParity(), eps=0.02, max_iter=30
        ).fit(
            self.pre_unaware.transform(Xtr.drop(columns="female")),
            self.ytr_biased,
            sensitive_features=self.atr,
        )
        return self

    # -- predictions on raw dataframes ---------------------------------------
    def score(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Baseline model's estimated probability that the applicant repays."""
        return self.baseline.predict_proba(self.pre.transform(X_raw))[:, 1]

    def baseline_decision(self, X_raw: pd.DataFrame) -> np.ndarray:
        return (self.score(X_raw) >= 0.5).astype(int)

    def approval_prob(self, X_raw: pd.DataFrame, n: int = 200) -> np.ndarray:
        """Chance the FIXED model approves (it mixes two thresholds for a thin band of scores)."""
        Xt = self.pre.transform(X_raw)
        a = X_raw["female"].to_numpy()
        rep = np.repeat(np.arange(len(a)), n)
        pred = self.fixed.predict(Xt[rep], sensitive_features=a[rep], random_state=self.seed)
        return pred.reshape(len(a), n).mean(axis=1)

    def fixed_decision(self, X_raw: pd.DataFrame) -> np.ndarray:
        return (self.approval_prob(X_raw) >= 0.5).astype(int)

    def thresholds(self) -> dict:
        d = self.fixed.interpolated_thresholder_.interpolation_dict
        out = {}
        for g, name in ((0, "male"), (1, "female")):
            p0, p1 = float(d[g]["p0"]), float(d[g]["p1"])
            t0 = float(d[g]["operation0"].threshold)
            t1 = float(d[g]["operation1"].threshold)
            out[name] = round(p0 * t0 + p1 * t1, 3)  # effective (mixed) cut-off
        return out


# ------------------------------------------------------------------------ metrics
def evaluate(y_true, y_pred, a) -> dict:
    """All fairness numbers for one set of decisions (measured against TRUE labels)."""
    mf = MetricFrame(
        metrics={"approval_rate": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=pd.Series(a).map({0: "Male", 1: "Female"}),
    )
    g = mf.by_group
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "by_group": {
            grp: {k: float(g.loc[grp, k]) for k in ("approval_rate", "tpr", "fpr")} for grp in ("Male", "Female")
        },
        # signed gaps, male - female (positive = women disadvantaged)
        "approval_gap": float(g.loc["Male", "approval_rate"] - g.loc["Female", "approval_rate"]),
        "equal_opportunity_gap": float(g.loc["Male", "tpr"] - g.loc["Female", "tpr"]),
        # Fairlearn's own (unsigned) headline metrics
        "demographic_parity_difference": float(demographic_parity_difference(y_true, y_pred, sensitive_features=a)),
        "equalized_odds_difference": float(equalized_odds_difference(y_true, y_pred, sensitive_features=a)),
    }


def predictions(m: FairLoanModels) -> dict:
    """Decisions of every model on the clean test set."""
    Xte_p = m.pre_unaware.transform(m.Xte_raw.drop(columns="female"))
    return {
        "Baseline": m.baseline.predict(m.Xte),
        "Fixed (Fairlearn)": m.fixed.predict(m.Xte, sensitive_features=m.ate, random_state=m.seed),
        "Baseline without gender column": m.unaware.predict(Xte_p),
        EG_NAME: m.eg.predict(Xte_p, random_state=m.seed),
        "Reference (trained on fair labels)": m.reference.predict(m.Xte),
    }


def evaluate_all(m: FairLoanModels) -> dict:
    return {name: evaluate(m.yte, pred, m.ate) for name, pred in predictions(m).items()}



# ------------------------------------------------- fairness dial and subgroup views
def _dial_rows(m: FairLoanModels) -> list:
    """Move the women's approval cut-off (men stay at 0.5) and record what happens."""
    sc, a, y = m.score(m.Xte_raw), m.ate, m.yte
    rows = []
    for cut in DIAL_CUTS:
        pred = np.where(a == 1, sc >= cut, sc >= 0.5).astype(int)
        rows.append(
            dict(
                cutoff=cut,
                accuracy=float(accuracy_score(y, pred)),
                approval_gap=float(pred[a == 0].mean() - pred[a == 1].mean()),
                equal_opportunity_gap=float(pred[(a == 0) & (y == 1)].mean() - pred[(a == 1) & (y == 1)].mean()),
                female_approval=float(pred[a == 1].mean()),
            )
        )
    return rows


_SUB_MODELS = {"Baseline": "baseline", "Fixed (Fairlearn)": "fixed", EG_NAME: "eg"}


def _pool_subgroups(store: dict, m: FairLoanModels, preds: dict) -> None:
    """Add this split's approvals to running totals per subgroup, model and gender."""
    for col in SUBGROUP_COLS:
        for val in m.Xte_raw[col].unique():
            mask = (m.Xte_raw[col] == val).to_numpy()
            for name, key in _SUB_MODELS.items():
                for g in (0, 1):
                    sel = mask & (m.ate == g)
                    cell = store.setdefault((col, str(val), key, g), [0, 0])
                    cell[0] += int(preds[name][sel].sum())
                    cell[1] += int(sel.sum())


def _subgroup_table(store: dict) -> list:
    df = load_data()
    keys = sorted({(c, v) for c, v, _, _ in store})
    out = []
    for col, val in keys:
        row = {
            "column": col,
            "value": val,
            "n_men": int(((df[col].astype(str) == val) & (df["female"] == 0)).sum()),
            "n_women": int(((df[col].astype(str) == val) & (df["female"] == 1)).sum()),
        }
        for key in ("baseline", "fixed", "eg"):
            rate = {}
            for g in (0, 1):
                ap, tot = store[(col, val, key, g)]
                rate[g] = ap / tot if tot else None
            row[key + "_gap"] = None if rate[0] is None or rate[1] is None else float(rate[0] - rate[1])
        out.append(row)
    return out

# ------------------------------------------------------------ multi-seed robustness
def multi_seed(bias: float, seeds=range(20), population: bool = False) -> dict:
    """Average over many random splits - the test set is small, so one split is noisy.

    population=True also averages the 'qualified women rejected' counts and the gender-flip test.
    """
    rows, pop, dial, sub = [], [], [], {}
    for s_ in seeds:
        m = FairLoanModels(seed=s_, bias=bias).fit()
        preds = predictions(m)
        for name, pred in preds.items():
            res = evaluate(m.yte, pred, m.ate)
            rows.append(
                dict(
                    model=name,
                    accuracy=res["accuracy"],
                    approval_gap=res["approval_gap"],
                    equal_opportunity_gap=res["equal_opportunity_gap"],
                    male_approval=res["by_group"]["Male"]["approval_rate"],
                    female_approval=res["by_group"]["Female"]["approval_rate"],
                    male_tpr=res["by_group"]["Male"]["tpr"],
                    female_tpr=res["by_group"]["Female"]["tpr"],
                    dp_diff=res["demographic_parity_difference"],
                    eo_diff=res["equalized_odds_difference"],
                )
            )
        if population:
            wr, cf = wrongly_rejected_counts(m), counterfactual_flip_rates(m)
            pop.append({**wr, **cf})
            dial.extend(_dial_rows(m))
            _pool_subgroups(sub, m, preds)
    df = pd.DataFrame(rows)
    mean, std = df.groupby("model").mean(), df.groupby("model").std()
    out = {
        "n_splits": len(list(seeds)),
        "mean": mean.round(4).to_dict("index"),
        "std": std.round(4).to_dict("index"),
    }
    if population:
        out["population"] = {k: float(v) for k, v in pd.DataFrame(pop).mean().items()}
        out["dial"] = pd.DataFrame(dial).groupby("cutoff").mean().round(4).to_dict("index")
        out["dial"] = {str(k): v for k, v in out["dial"].items()}
        out["subgroups"] = _subgroup_table(sub)
    return out


# ------------------------------------------------------------ the individual story
FRIENDLY = {
    "Married": "Married",
    "Dependents": "Dependents",
    "Education": "Education",
    "Self_Employed": "Self-employed",
    "ApplicantIncome": "Applicant income (₹ per month)",
    "CoapplicantIncome": "Co-applicant income (₹ per month)",
    "LoanAmount": "Loan amount (₹ thousand)",
    "Loan_Amount_Term": "Loan term (months)",
    "Credit_History": "Credit history (1 = good)",
    "Property_Area": "Property area",
}


def find_individual(m: FairLoanModels) -> dict:
    """A woman who truly repays, was rejected by the baseline, and is approved after the fix.
    Among such women we show the one whose baseline score rises most when ONLY her gender changes.

    Also runs the counterfactual: change ONLY her gender and see whether each model changes its mind.
    """
    Xte = m.Xte_raw.reset_index(drop=True)
    women = np.where((m.ate == 1) & (m.yte == 1))[0]
    Xw = Xte.iloc[women]
    Xw_male = Xw.assign(female=0)

    base_dec = m.baseline_decision(Xw)
    base_dec_male = m.baseline_decision(Xw_male)
    fix_prob = m.approval_prob(Xw)
    fix_prob_male = m.approval_prob(Xw_male)
    score = m.score(Xw)

    cand = np.where((base_dec == 0) & (fix_prob == 1.0) & (base_dec_male == 1) & (fix_prob_male == 1.0))[0]
    if len(cand) == 0:  # relax: skip the counterfactual requirement
        cand = np.where((base_dec == 0) & (fix_prob == 1.0))[0]
    if len(cand) == 0:
        return {}
    jump = m.score(Xw_male) - score            # how much the score rises if only gender changes
    order = cand[np.argsort(-jump[cand])]
    i = int(order[0])
    row = Xw.iloc[[i]]
    profile = {FRIENDLY[k]: (row[k].iloc[0].item() if hasattr(row[k].iloc[0], "item") else row[k].iloc[0]) for k in FRIENDLY}
    return {
        "test_row": int(women[i]),
        "profile": profile,
        "true_outcome": "Would repay (good credit risk)",
        "baseline": {"score": float(score[i]), "decision": "REJECTED" if base_dec[i] == 0 else "APPROVED"},
        "baseline_if_male": {
            "score": float(m.score(Xw_male.iloc[[i]])[0]),
            "decision": "APPROVED" if base_dec_male[i] == 1 else "REJECTED",
        },
        "fixed": {"approval_probability": float(fix_prob[i]), "decision": "APPROVED" if fix_prob[i] >= 0.5 else "REJECTED"},
        "fixed_if_male": {
            "approval_probability": float(fix_prob_male[i]),
            "decision": "APPROVED" if fix_prob_male[i] >= 0.5 else "REJECTED",
        },
        "n_candidates": int(len(cand)),
        "alternatives": [int(women[j]) for j in order[1:4]],
    }


def counterfactual_flip_rates(m: FairLoanModels) -> dict:
    """Among all women in the test set: how often does changing ONLY gender flip the decision?"""
    Xte = m.Xte_raw.reset_index(drop=True)
    Xw = Xte[m.ate == 1]
    Xw_male = Xw.assign(female=0)
    b = (m.baseline_decision(Xw) != m.baseline_decision(Xw_male)).mean()
    f = (m.fixed_decision(Xw) != m.fixed_decision(Xw_male)).mean()
    up = ((m.baseline_decision(Xw) == 0) & (m.baseline_decision(Xw_male) == 1)).mean()
    return {"baseline_flip_rate": float(b), "fixed_flip_rate": float(f), "baseline_rejected_but_approved_as_male": float(up)}


def wrongly_rejected_counts(m: FairLoanModels) -> dict:
    """How many truly-good women each model rejects on the test set."""
    good_f = (m.ate == 1) & (m.yte == 1)
    pr = predictions(m)
    return {
        "n_good_women_test": int(good_f.sum()),
        "baseline_rejected": int((pr["Baseline"][good_f] == 0).sum()),
        "fixed_rejected": int((pr["Fixed (Fairlearn)"][good_f] == 0).sum()),
    }


def proxy_leakage(seed: int = SHOWCASE_SEED) -> dict:
    """Can gender be guessed from the OTHER columns? If yes, deleting the column can't remove the bias."""
    X, _, a = split_xy(load_data())
    Xn = X.drop(columns="female")
    pre = make_preprocessor(Xn, use_sex=True)
    Z = pre.fit_transform(Xn)
    p = cross_val_predict(LogisticRegression(max_iter=3000, C=0.5), Z, a, cv=5, method="predict_proba")[:, 1]
    lr = LogisticRegression(max_iter=3000, C=0.5).fit(Z, a)
    names = pre.get_feature_names_out()
    coefs = pd.Series(lr.coef_[0], index=names).sort_values(key=np.abs, ascending=False).head(6)
    return {
        "auc_predicting_sex_from_other_columns": float(roc_auc_score(a, p)),
        "top_proxy_features": {k.replace("cat__", "").replace("num__", ""): round(float(v), 3) for k, v in coefs.items()},
    }


# ---------------------------------------------------------------------- build all
def build_results(seeds=range(20), sweep=(0.0, 0.1, 0.2, 0.3, 0.4)) -> tuple[dict, FairLoanModels]:
    m = FairLoanModels(seed=SHOWCASE_SEED, bias=DEFAULT_BIAS).fit()
    df = load_data()
    results = {
        "config": {
            "dataset": "Dream Housing Finance loan data (India), " + str(len(df)) + " applicants with a recorded gender",
            "sensitive_attribute": "gender (Male / Female)",
            "injected_bias_share": DEFAULT_BIAS,
            "showcase_seed": SHOWCASE_SEED,
            "n_train": int(len(m.ytr_true)),
            "n_test": int(len(m.yte)),
            "n_women_test": int((m.ate == 1).sum()),
            "share_women": float(df["female"].mean()),
            "n_women_total": int(df["female"].sum()),
            "n_dropped_no_gender": int(pd.read_csv(DATA_PATH)[SENSITIVE_SOURCE].isna().sum()),
            "raw_approval_men": float(df.loc[df["female"] == 0, "credit_risk"].mean()),
            "raw_approval_women": float(df.loc[df["female"] == 1, "credit_risk"].mean()),
            "labels_flipped": int((m.ytr_biased != m.ytr_true).sum()),
            "train_good_rate_men": float(m.ytr_biased[m.atr == 0].mean()),
            "train_good_rate_women": float(m.ytr_biased[m.atr == 1].mean()),
            "true_good_rate_men": float(m.ytr_true[m.atr == 0].mean()),
            "true_good_rate_women": float(m.ytr_true[m.atr == 1].mean()),
        },
        "showcase": evaluate_all(m),
        "thresholds": m.thresholds(),
        "individual": find_individual(m),
        "counterfactual": counterfactual_flip_rates(m),
        "wrongly_rejected": wrongly_rejected_counts(m),
        "proxy": proxy_leakage(),
        "multi_seed": multi_seed(DEFAULT_BIAS, seeds, population=True),
        "sweep": {str(p): multi_seed(p, range(10)) for p in sweep},
    }
    return results, m
