"""FairLoan - catch it, fix it, prove it.   Run:  streamlit run app.py"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import pipeline as pl  # noqa: F401  (needed to unpickle the models)

ROOT = Path(__file__).parent
MALE, FEMALE = "#2a78d6", "#eb6834"          # validated categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2de"
BASE_C, UNAWARE_C, FIXED_C = "#52514e", "#9a9994", "#1baf7a"

st.set_page_config(page_title="FairLoan", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1200px;}
div[data-testid="stMetric"] {background:#ffffff; border:1px solid #e3e2de; border-radius:14px;
    padding:14px 18px; box-shadow:0 1px 3px rgba(0,0,0,.05);}
div[data-testid="stMetricValue"] {font-weight:700;}
button[data-baseweb="tab"] {font-size:1.02rem; font-weight:600;}
.hero {background:linear-gradient(135deg,#0f2a4a 0%,#2a78d6 100%); color:#fff; border-radius:18px;
    padding:26px 30px; margin-bottom:18px;}
.hero h1 {color:#fff; margin:0 0 6px 0; font-size:2.1rem; padding:0;}
.hero p {color:#dbe8fa; margin:0; font-size:1rem;}
.steps {display:flex; gap:10px; margin-top:16px; flex-wrap:wrap;}
.step {background:rgba(255,255,255,.14); border-radius:999px; padding:6px 16px; font-weight:600; font-size:.92rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load():
    res = json.loads((ROOT / "artifacts" / "results.json").read_text())
    models = joblib.load(ROOT / "artifacts" / "models.joblib")
    return res, models


R, M = load()
MS = R["multi_seed"]["mean"]
SD = R["multi_seed"]["std"]
B, F, U = "Baseline", "Fixed (Fairlearn)", "Baseline without gender column"
pct = lambda x: f"{100 * x:.0f}%"
pts = lambda x: f"{100 * x:.0f} pts"


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def grouped_bars(models, male, female, ylabel, title):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(models)); w = 0.36
    for off, vals, col, lab in ((-w / 2 - .01, male, MALE, "Men"), (w / 2 + .01, female, FEMALE, "Women")):
        bars = ax.bar(x + off, vals, w, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + .015, pct(v), ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x, models, fontsize=10, color=INK)
    ax.set_ylim(0, 1.22); ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10); ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=10, labelcolor=INK)
    style(ax); fig.tight_layout()
    return fig


st.markdown(
    f"""<div class="hero"><h1>⚖️ FairLoan: when the training data has a bias</h1>
    <p>{R['config']['dataset']} · Fairlearn · every number is measured on a clean test set</p>
    <div class="steps"><span class="step">① Catch the bias</span><span class="step">② Fix it</span><span class="step">③ Prove it on one person</span></div></div>""",
    unsafe_allow_html=True,
)

t1, t2, t3, t4, t5 = st.tabs(["① Catch it", "② Fix it", "③ Prove it (one person)", "④ Try it yourself", "Honest limits"])

# ------------------------------------------------------------------ 1. CATCH
with t1:
    c = R["config"]
    st.subheader("A normal loan model quietly learns the unfairness in its history")
    st.markdown(
        f"We took real loan data and recreated the problem on purpose: in the **training** history, "
        f"**{pct(c['injected_bias_share'])} of women who really did repay** were recorded as defaults "
        f"({c['labels_flipped']} labels flipped, nothing else changed). Then we trained an ordinary logistic-regression model. "
        f"The **test** labels stay clean, so we know exactly who deserved a loan."
    )
    a, b, cc, d = st.columns(4)
    a.metric("Men approved", pct(MS[B]["male_approval"]))
    b.metric("Women approved", pct(MS[B]["female_approval"]), f"-{pts(MS[B]['approval_gap'])} vs men")
    cc.metric("Qualified men approved", pct(MS[B]["male_tpr"]))
    d.metric("Qualified women approved", pct(MS[B]["female_tpr"]), f"-{pts(MS[B]['equal_opportunity_gap'])} vs men")
    l, r = st.columns(2)
    with l:
        st.pyplot(grouped_bars(["Baseline"], [MS[B]["male_approval"]], [MS[B]["female_approval"]], "share approved", "Everyone: approval rate"))
    with r:
        st.pyplot(grouped_bars(["Baseline"], [MS[B]["male_tpr"]], [MS[B]["female_tpr"]], "share approved", "Equally qualified (would repay): approval rate"))
    st.info(
        f"**How to read it.** The right chart is the damning one: among applicants who *would have repaid*, the model approves "
        f"{pct(MS[B]['male_tpr'])} of men but only {pct(MS[B]['female_tpr'])} of women — a {pts(MS[B]['equal_opportunity_gap'])} gap "
        f"(average of {R['multi_seed']['n_splits']} random splits, ±{pts(SD[B]['equal_opportunity_gap'])} std). "
        f"Nobody told the model to discriminate; it copied the history."
    )
    with st.expander("Fairlearn metrics for the baseline (single showcase split)"):
        s = R["showcase"][B]
        st.table(pd.DataFrame({
            "Metric": ["Accuracy vs true labels", "Demographic parity difference", "Equal opportunity gap (TPR)", "Equalized odds difference"],
            "Value": [f"{s['accuracy']:.3f}", f"{s['demographic_parity_difference']:.3f}", f"{s['equal_opportunity_gap']:.3f}", f"{s['equalized_odds_difference']:.3f}"],
        }))

# ------------------------------------------------------------------- 2. FIX
with t2:
    st.subheader("Fix: keep the model, adjust how its scores turn into decisions")
    th = R["thresholds"]
    st.markdown(
        "We use Fairlearn's **ThresholdOptimizer** with a *demographic parity* constraint. It leaves the trained model alone and "
        "learns one approval cut-off per group so both groups are approved at the same rate, while losing as little accuracy as possible. "
        f"Learned cut-offs on the model's repay-probability: **men ≥ {th['male']:.2f}**, **women ≥ {th['female']:.2f}** "
        "— it compensates for the score penalty the biased history put on women."
    )
    order = [B, U, F]
    labels = ["Baseline", "Delete gender\ncolumn", "Fixed\n(Fairlearn)"]
    l, r = st.columns(2)
    with l:
        st.pyplot(grouped_bars(labels, [MS[k]["male_approval"] for k in order], [MS[k]["female_approval"] for k in order], "share approved", "Approval rate, everyone"))
    with r:
        st.pyplot(grouped_bars(labels, [MS[k]["male_tpr"] for k in order], [MS[k]["female_tpr"] for k in order], "share approved", "Approval rate, equally qualified"))

    st.markdown("**Accuracy vs. fairness** (mean of 20 random splits; accuracy uses the clean labels)")
    ref = "Reference (trained on fair labels)"
    tbl = pd.DataFrame({
        "Model": ["Baseline (biased history)", "Delete the gender column", "Fixed (Fairlearn)", "Reference: trained on fair labels"],
        "Accuracy": [MS[k]["accuracy"] for k in (B, U, F, ref)],
        "Approval gap (M−F)": [MS[k]["approval_gap"] for k in (B, U, F, ref)],
        "Qualified-approval gap (M−F)": [MS[k]["equal_opportunity_gap"] for k in (B, U, F, ref)],
    }).set_index("Model")
    st.dataframe(tbl.style.format("{:.3f}"), width="stretch")
    st.caption("The reference row is what we'd get if history had been fair — it isn't 0 because the raw data has a small natural gap between men and women.")

    st.markdown("**Does it hold up as the historical bias gets worse?**")
    ps = sorted(R["sweep"], key=float)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for key, col, ls, lab in ((B, BASE_C, "-", "Baseline"), (U, UNAWARE_C, "--", "Delete gender column"), (F, FIXED_C, "-", "Fixed (Fairlearn)")):
        ys = [R["sweep"][p]["mean"][key]["approval_gap"] for p in ps]
        xs = [float(p) for p in ps]
        ax.plot(xs, ys, color=col, ls=ls, lw=2, marker="o", ms=6, mfc=col, mec="#fcfcfb", mew=2)
        ax.text(xs[-1] + .01, ys[-1], lab, color=INK, va="center", fontsize=10)
    ax.set_xlim(-.01, .55); ax.set_xticks([float(p) for p in ps], [pct(float(p)) for p in ps])
    ax.set_xlabel("share of qualified women wrongly marked 'default' in the training history", color=MUTED, fontsize=9)
    ax.set_ylabel("approval gap, men − women", color=MUTED, fontsize=10)
    ax.set_yticks([0, .2, .4, .6], ["0", "20 pts", "40 pts", "60 pts"]); style(ax); fig.tight_layout()
    st.pyplot(fig)
    st.success(
        f"Approval gap {pts(MS[B]['approval_gap'])} → **{pts(MS[F]['approval_gap'])}**, qualified-approval gap "
        f"{pts(MS[B]['equal_opportunity_gap'])} → **{pts(abs(MS[F]['equal_opportunity_gap']))}**, and accuracy against the true labels "
        f"{MS[B]['accuracy']:.3f} → {MS[F]['accuracy']:.3f}: fairer *and* no less accurate, because the biased labels were making the baseline wrong about women."
    )
    st.warning(
        "**Be straight about this:** deleting the gender column also closes most of the gap on this dataset, because the other columns barely hint at gender "
        f"(gender can be guessed from them with AUC {R['proxy']['auc_predicting_sex_from_other_columns']:.2f}, where 0.50 is a coin flip). "
        "What Fairlearn adds is a *measurable, tunable guarantee* (equal approval rates) that doesn't depend on how strong such hints are, and it works when you can't or shouldn't drop a column."
    )

# ------------------------------------------------------------------ 3. PROVE
with t3:
    ind = R["individual"]
    pop = R["multi_seed"]["population"]
    st.subheader("One real applicant from the test set")
    if ind:
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("**Her application**")
            st.table(pd.DataFrame({"Value": {k: str(v) for k, v in ind["profile"].items()}}))
            st.markdown("Ground truth: **" + ind["true_outcome"] + "** — she was never in the training data.")
        with right:
            st.markdown("**What each model decided**")
            x1, x2 = st.columns(2)
            x1.metric("Baseline model", ind["baseline"]["decision"]); x1.caption(f"repay score {ind['baseline']['score']:.2f} (needs 0.50)")
            x2.metric("Fixed model", ind["fixed"]["decision"]); x2.caption("same person, same data")
            st.markdown("**Counterfactual: change only her gender, nothing else**")
            y1, y2 = st.columns(2)
            y1.metric("Baseline, if she were male", ind["baseline_if_male"]["decision"]); y1.caption(f"score {ind['baseline_if_male']['score']:.2f}")
            y2.metric("Fixed, if she were male", ind["fixed_if_male"]["decision"])
            y2.caption("decision unchanged" if ind["fixed_if_male"]["decision"] == ind["fixed"]["decision"] else "decision changes")
            if ind["baseline_if_male"]["decision"] == "APPROVED":
                st.error(
                    f"The baseline rejects her at {ind['baseline']['score']:.2f} but would approve the identical applicant as a man "
                    f"({ind['baseline_if_male']['score']:.2f}). After the fix she is approved."
                )
            if ind["fixed_if_male"]["decision"] != ind["fixed"]["decision"]:
                st.info("The fixed model uses a separate cut-off for each group by design, so it does not treat the two versions identically. "
                        "That is the trade-off flagged on the Honest limits tab.")
        st.caption(
            f"Illustration only. Taken from a typical split (chosen because its numbers are close to the 20-split average). Among women who truly repay, "
            f"are rejected by the baseline and approved by the fixed model, this is the one whose score rises most when only gender changes "
            f"({ind['n_candidates']} such women in this split)."
        )
    st.markdown("---")
    st.subheader("It isn't just one person")
    a, b, c3 = st.columns(3)
    n_good = pop["n_good_women_test"]
    a.metric("Qualified women rejected (baseline)", f"{pop['baseline_rejected']:.1f} of {n_good:.0f}")
    b.metric("Qualified women rejected (fixed)", f"{pop['fixed_rejected']:.1f} of {n_good:.0f}", f"{pop['fixed_rejected'] - pop['baseline_rejected']:.1f}", delta_color="inverse")
    c3.metric("Decisions that flip if only gender changes", f"{pct(pop['baseline_flip_rate'])} → {pct(pop['fixed_flip_rate'])}")
    st.caption(
        f"Average of {R['multi_seed']['n_splits']} random splits (about {n_good:.0f} qualified women in each test set). "
        "Gender-flip test: take every woman in the test set, change only her gender, and count how often the decision changes."
    )

# -------------------------------------------------------------- 4. TRY IT
with t4:
    st.subheader("Try it yourself")
    st.markdown("Edit an applicant. The same person is always scored as a woman **and** as a man, by both models.")
    df = pl.load_data()
    X_all = pl.split_xy(df)[0]
    fields = [c for c in X_all.columns if c != "female"]
    cols = st.columns(3)
    row = {}
    for i, c in enumerate(fields):          # widgets are built from the data, so any dataset works
        box, label = cols[i % 3], pl.FRIENDLY.get(c, c)
        if pd.api.types.is_numeric_dtype(X_all[c]) and X_all[c].nunique() > 6:
            row[c] = box.number_input(label, int(X_all[c].min()), int(X_all[c].max()), int(X_all[c].median()))
        else:
            options = sorted(X_all[c].unique())
            row[c] = box.selectbox(label, options, index=options.index(X_all[c].mode().iloc[0]))
    person = pd.DataFrame([row])[fields]
    rows = []
    for who, flag in (("as a woman", 1), ("as a man", 0)):
        p = person.assign(female=flag)
        rows.append({"Same applicant…": who,
                     "Repay score (baseline)": f"{M.score(p)[0]:.2f}",
                     "Baseline decision": "✅ Approved" if M.baseline_decision(p)[0] else "❌ Rejected",
                     "Fixed decision": "✅ Approved" if M.fixed_decision(p)[0] else "❌ Rejected"})
    st.table(pd.DataFrame(rows).set_index("Same applicant…"))
    st.caption("Starting values are the most common ones in the data. Demo only — not a real credit decision.")

# ---------------------------------------------------------------- 5. LIMITS
with t5:
    c = R["config"]
    st.subheader("What this does and doesn't show")
    st.markdown(f"""
- **The bias is injected, on purpose.** In the raw data women were approved {pct(c['raw_approval_women'])} of the time and men {pct(c['raw_approval_men'])}, almost the same. So the unfairness this demo shows is the one we put in. We can only *prove* a model copied unfairness if we know the fair answer, so we corrupted training labels ourselves and kept test labels clean. Real datasets rarely let you verify this; there you can only measure outcome gaps.
- **Small data.** {c['n_train'] + c['n_test']} applicants, of whom {c['n_women_total']} are women; about {c['n_women_test']} women in each test set. That is why headline numbers average 20 random splits, and single-split numbers move around (baseline qualified-approval gap: ±{pts(SD[B]['equal_opportunity_gap'])}).
- **Public Indian-style data, not real bureau data.** Real CIBIL data is private. This is a public loan-eligibility dataset from an Indian housing-finance setting; check its licence before publishing.
- **Fairness metrics conflict.** We equalise approval rates (demographic parity). Equal error rates (equalized odds) can't be fully satisfied at the same time; our fixed model still has an error-rate gap (equalized-odds difference {MS[F]['eo_diff']:.2f}).
- **Uses gender at decision time.** ThresholdOptimizer needs the group label when scoring, so it applies a different cut-off to each group; that is restricted in some jurisdictions. Fairlearn's in-training `ExponentiatedGradient` avoids that, at some cost in stability.
- **Gender is a stand-in.** It is recorded only as Male or Female, and the {c['n_dropped_no_gender']} applicants with no recorded gender were dropped rather than guessed. Marital status, education, urban or rural area and combinations of these are worth auditing too.
- **Deleting the gender column** already closes most of the gap here, because the other columns barely hint at gender. In data with strong hints such as pin code it would help much less; we did not test that.
""")
