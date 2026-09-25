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

st.set_page_config(page_title="FairLoan", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
h1, h2, h3 {font-family: Georgia, 'Times New Roman', serif; font-weight: 600; letter-spacing: -0.01em;}
h1 {font-size: 2.3rem !important; margin-bottom: 0.1rem;}
div[data-testid="stMetric"] {border-left: 3px solid #d9d8d3; padding: 2px 0 2px 14px;}
div[data-testid="stMetricValue"] {font-weight: 600;}
button[data-baseweb="tab"] {font-size: 1rem;}
.sub {color:#52514e; margin: 0 0 0.8rem 0; font-size: 1.02rem;}
hr.thin {border:0; border-top:1px solid #e3e2de; margin: 0.4rem 0 1rem 0;}
.note {border-left: 3px solid #2a78d6; padding: 6px 14px; background:#f4f6f9; margin: 10px 0;}
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
E = "Fairlearn in-training (no gender at decision time)"
pct = lambda x: f"{100 * x:.0f}%"
pts = lambda x: f"{100 * x:.0f} pts"


def style(ax):
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def grouped_bars(models, male, female, ylabel, title):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(models)); w = 0.36
    for off, vals, col, lab in ((-w / 2 - .01, male, MALE, "Men"), (w / 2 + .01, female, FEMALE, "Women")):
        bars = ax.bar(x + off, vals, w, color=col, label=lab)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + .015, pct(v), ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x, models, fontsize=10, color=INK)
    ax.set_ylim(0, 1.22); ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10); ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=10, labelcolor=INK)
    style(ax); fig.tight_layout()
    return fig


def audit_report():
    """Plain-text audit report built from the stored results, offered as a download."""
    c = R["config"]; pop = R["multi_seed"]["population"]; th = R["thresholds"]
    return f"""FAIRLOAN BIAS AUDIT REPORT
Data: {c['dataset']}
Group tested: gender (Male / Female)
Results are the average of {R['multi_seed']['n_splits']} random train/test splits, measured against clean test labels.

1. MODEL TESTED
Logistic regression trained on a history in which {pct(c['injected_bias_share'])} of qualified women were recorded as rejected
({c['labels_flipped']} labels changed in the showcase split). Test labels were not changed.

2. BIAS MEASURED (before the fix)
Approval rate, men / women:               {pct(MS[B]['male_approval'])} / {pct(MS[B]['female_approval'])}   (gap {pts(MS[B]['approval_gap'])})
Approval rate among qualified, men / women: {pct(MS[B]['male_tpr'])} / {pct(MS[B]['female_tpr'])}   (gap {pts(MS[B]['equal_opportunity_gap'])}, spread +/-{pts(SD[B]['equal_opportunity_gap'])})
Decisions that change if only gender is changed: {pct(pop['baseline_flip_rate'])}
Qualified women rejected: {pop['baseline_rejected']:.1f} of {pop['n_good_women_test']:.0f} per test set
Accuracy vs true labels: {MS[B]['accuracy']:.3f}

3. FIX APPLIED
Fairlearn ThresholdOptimizer, demographic parity constraint.
Approval cut-offs learned: men {th['male']:.2f}, women {th['female']:.2f}

4. RE-TEST (after the fix)
Approval rate, men / women:               {pct(MS[F]['male_approval'])} / {pct(MS[F]['female_approval'])}   (gap {pts(MS[F]['approval_gap'])})
Approval rate among qualified, men / women: {pct(MS[F]['male_tpr'])} / {pct(MS[F]['female_tpr'])}   (gap {pts(abs(MS[F]['equal_opportunity_gap']))})
Decisions that change if only gender is changed: {pct(pop['fixed_flip_rate'])}
Qualified women rejected: {pop['fixed_rejected']:.1f} of {pop['n_good_women_test']:.0f} per test set
Accuracy vs true labels: {MS[F]['accuracy']:.3f}

5. COMPARISON POINTS
Deleting the gender column instead: approval gap {pts(MS[U]['approval_gap'])}, accuracy {MS[U]['accuracy']:.3f}.
Fairlearn ExponentiatedGradient (no gender needed at decision time): approval gap {pts(MS[E]['approval_gap'])}, accuracy {MS[E]['accuracy']:.3f}.

6. LIMITATIONS
- The bias in this test was injected on purpose; the raw data shows women approved {pct(c['raw_approval_women'])} and men {pct(c['raw_approval_men'])} of the time.
- Small sample ({c['n_train'] + c['n_test']} applicants); numbers vary between splits.
- The label is a recorded decision, not a confirmed repayment outcome.
- The threshold fix uses gender at decision time and applies a separate cut-off per group. The in-training fix does not.
- Equal approval rates do not give equal error rates (equalized-odds difference {MS[F]['eo_diff']:.2f}).
"""


st.title("FairLoan")
st.markdown(f"<p class='sub'>When the training data has a bias: catch it, fix it, prove it. "
            f"{R['config']['dataset']}. Every number is measured on a clean test set.</p><hr class='thin'>", unsafe_allow_html=True)

t1, t2, t3, t7, t4, t5, t6 = st.tabs(["Catch", "Fix", "Prove", "Subgroups", "Try it", "Impact and roadmap", "Limits"])

# ------------------------------------------------------------------ 1. CATCH
with t1:
    c = R["config"]
    st.subheader("A normal loan model learns the unfairness in its history")
    st.markdown(
        f"We took real loan data and recreated the problem on purpose. In the **training** history, "
        f"**{pct(c['injected_bias_share'])} of the women who were approved** were recorded as rejected instead "
        f"({c['labels_flipped']} labels changed, nothing else). We then trained an ordinary logistic-regression model. "
        f"The **test** labels stay clean, so we know exactly who was qualified."
    )
    a, b, cc, d = st.columns(4)
    a.metric("Men approved", pct(MS[B]["male_approval"]))
    b.metric("Women approved", pct(MS[B]["female_approval"]))
    b.caption(f"{pts(MS[B]['approval_gap'])} below men")
    cc.metric("Qualified men approved", pct(MS[B]["male_tpr"]))
    d.metric("Qualified women approved", pct(MS[B]["female_tpr"]))
    d.caption(f"{pts(MS[B]['equal_opportunity_gap'])} below men")
    l, r = st.columns(2)
    with l:
        st.pyplot(grouped_bars(["Baseline"], [MS[B]["male_approval"]], [MS[B]["female_approval"]], "share approved", "Everyone: approval rate"))
    with r:
        st.pyplot(grouped_bars(["Baseline"], [MS[B]["male_tpr"]], [MS[B]["female_tpr"]], "share approved", "Qualified applicants only: approval rate"))
    st.markdown(
        f"<div class='note'>The right-hand chart is the one that matters. Among applicants who were qualified, the model approves "
        f"{pct(MS[B]['male_tpr'])} of men but only {pct(MS[B]['female_tpr'])} of women, a gap of {pts(MS[B]['equal_opportunity_gap'])} "
        f"(average of {R['multi_seed']['n_splits']} random splits, standard deviation {pts(SD[B]['equal_opportunity_gap'])}). "
        f"Nobody told the model to discriminate. It copied the history.</div>", unsafe_allow_html=True)
    with st.expander("Fairlearn metrics for the baseline (single showcase split)"):
        s = R["showcase"][B]
        st.table(pd.DataFrame({
            "Metric": ["Accuracy vs true labels", "Demographic parity difference", "Equal opportunity gap (TPR)", "Equalized odds difference"],
            "Value": [f"{s['accuracy']:.3f}", f"{s['demographic_parity_difference']:.3f}", f"{s['equal_opportunity_gap']:.3f}", f"{s['equalized_odds_difference']:.3f}"],
        }))

# ------------------------------------------------------------------- 2. FIX
with t2:
    st.subheader("The fix: keep the model, change where the approval line sits")
    th = R["thresholds"]
    st.markdown(
        "We use Fairlearn's **ThresholdOptimizer** with a demographic parity constraint. It leaves the trained model alone and "
        "learns one approval cut-off for each group, so both groups are approved at the same rate while accuracy suffers as little as possible. "
        f"The learned cut-offs on the model's approval score are **{th['male']:.2f} for men** and **{th['female']:.2f} for women**. "
        "The lower line for women offsets the penalty that the biased history put on their scores."
    )
    order = [B, U, F, E]
    labels = ["Baseline", "Delete\ngender", "Fixed\n(threshold)", "In-training\n(no gender)"]
    l, r = st.columns(2)
    with l:
        st.pyplot(grouped_bars(labels, [MS[k]["male_approval"] for k in order], [MS[k]["female_approval"] for k in order], "share approved", "Approval rate, everyone"))
    with r:
        st.pyplot(grouped_bars(labels, [MS[k]["male_tpr"] for k in order], [MS[k]["female_tpr"] for k in order], "share approved", "Approval rate, qualified applicants"))

    st.markdown("**Accuracy against fairness** (mean of 20 random splits; accuracy is measured on the clean labels)")
    ref = "Reference (trained on fair labels)"
    tbl = pd.DataFrame({
        "Model": ["Baseline (biased history)", "Delete the gender column", "Fixed: ThresholdOptimizer (needs gender at decision time)",
                  "Fixed: ExponentiatedGradient (no gender at decision time)", "Reference: trained on fair labels"],
        "Accuracy": [MS[k]["accuracy"] for k in (B, U, F, E, ref)],
        "Approval gap (M-F)": [MS[k]["approval_gap"] for k in (B, U, F, E, ref)],
        "Qualified-approval gap (M-F)": [MS[k]["equal_opportunity_gap"] for k in (B, U, F, E, ref)],
    }).set_index("Model")
    st.dataframe(tbl.style.format("{:.3f}"), width="stretch")
    st.caption("The reference row is what we would get if the history had been fair. It is not zero because the raw data has a small natural gap between men and women.")

    st.markdown(
        f"<div class='note'>The last fixed row matters for the legal question. ExponentiatedGradient uses gender only while training. "
        f"The finished model never sees it, so it needs no gender at decision time. On this data it reaches an approval gap of "
        f"{pts(MS[E]['approval_gap'])} at accuracy {MS[E]['accuracy']:.3f}, similar to the threshold fix.</div>", unsafe_allow_html=True)

    st.markdown("**Fairness dial: move the women's approval cut-off yourself**")
    st.caption("Men stay at the usual 0.50 line. Drag the women's line down and watch the gaps close. 0.50 is the untouched baseline. "
               f"Average of {R['multi_seed']['n_splits']} random splits.")
    dial = R["multi_seed"]["dial"]
    cuts = sorted(float(k) for k in dial)
    cut = st.slider("Women's approval cut-off", min(cuts), max(cuts), 0.50, 0.02, format="%.2f")
    d0 = dial[min(dial, key=lambda k: abs(float(k) - cut))]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Women approved", pct(d0["female_approval"]))
    m2.metric("Approval gap (men minus women)", pts(d0["approval_gap"]))
    m3.metric("Qualified-approval gap", pts(d0["equal_opportunity_gap"]))
    m4.metric("Accuracy", f"{d0['accuracy']:.3f}")
    figd, axd = plt.subplots(figsize=(7.5, 3.0))
    kk = lambda c_: str(round(c_, 2))
    axd.plot(cuts, [dial[kk(c_)]["approval_gap"] for c_ in cuts], color=BASE_C, lw=2, label="Approval gap")
    axd.plot(cuts, [dial[kk(c_)]["equal_opportunity_gap"] for c_ in cuts], color=FIXED_C, lw=2, label="Qualified-approval gap")
    axd.axvline(cut, color=FEMALE, lw=1.5, ls="--")
    axd.invert_xaxis()
    axd.set_xlabel("women's approval cut-off (lower means more women approved)", color=MUTED, fontsize=9)
    axd.set_yticks([0, .1, .2, .3, .4, .5], ["0", "10 pts", "20 pts", "30 pts", "40 pts", "50 pts"])
    axd.legend(frameon=False, fontsize=9, labelcolor=INK); style(axd); figd.tight_layout()
    st.pyplot(figd)
    st.caption(f"ThresholdOptimizer found its own answer without a slider: about {th['female']:.2f} for women against {th['male']:.2f} for men, "
               "using a slightly different method. The dial shows why a lender would need to decide where to stop: a lower line closes the gap but approves more people overall.")

    st.markdown("**Does it hold as the historical bias gets worse?**")
    ps = sorted(R["sweep"], key=float)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for key, col, ls, lab in ((B, BASE_C, "-", "Baseline"), (U, UNAWARE_C, "--", "Delete gender column"), (F, FIXED_C, "-", "Fixed (threshold)"), (E, "#7a4fc1", ":", "In-training")):
        ys = [R["sweep"][p]["mean"][key]["approval_gap"] for p in ps]
        xs = [float(p) for p in ps]
        ax.plot(xs, ys, color=col, ls=ls, lw=2, marker="o", ms=6, mfc=col, mec="#fcfcfb", mew=2, label=lab)
    ax.legend(frameon=False, fontsize=10, labelcolor=INK, loc="upper left")
    ax.set_xlim(-.01, .41); ax.set_xticks([float(p) for p in ps], [pct(float(p)) for p in ps])
    ax.set_xlabel("share of qualified women wrongly marked as rejected in the training history", color=MUTED, fontsize=9)
    ax.set_ylabel("approval gap, men minus women", color=MUTED, fontsize=10)
    ax.set_yticks([0, .2, .4, .6], ["0", "20 pts", "40 pts", "60 pts"]); style(ax); fig.tight_layout()
    st.pyplot(fig)
    st.markdown(
        f"<div class='note'>The approval gap falls from {pts(MS[B]['approval_gap'])} to <b>{pts(MS[F]['approval_gap'])}</b> and the qualified-approval gap "
        f"from {pts(MS[B]['equal_opportunity_gap'])} to <b>{pts(abs(MS[F]['equal_opportunity_gap']))}</b>. Accuracy against the true labels goes from "
        f"{MS[B]['accuracy']:.3f} to {MS[F]['accuracy']:.3f}. It does not drop, because the biased labels were making the baseline wrong about women.</div>",
        unsafe_allow_html=True)
    st.warning(
        "Deleting the gender column also closes most of the gap on this dataset, because the other columns barely hint at gender "
        f"(gender can be guessed from them with an AUC of {R['proxy']['auc_predicting_sex_from_other_columns']:.2f}, where 0.50 is a coin flip). "
        "What Fairlearn adds is a measurable guarantee of equal approval rates that does not depend on how strong such hints are, "
        "and it works when you cannot or should not drop a column."
    )

# ------------------------------------------------------------------ 3. PROVE
with t3:
    ind = R["individual"]
    pop = R["multi_seed"]["population"]
    st.subheader("One applicant from the test set")
    if ind:
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("**Her application**")
            st.table(pd.DataFrame({"Value": {k: str(v) for k, v in ind["profile"].items()}}))
            st.markdown("Original label: **approved (qualified)**. She was not in the training data.")
        with right:
            st.markdown("**What each model decided**")
            x1, x2 = st.columns(2)
            x1.metric("Baseline model", ind["baseline"]["decision"]); x1.caption(f"approval score {ind['baseline']['score']:.2f}, needs 0.50")
            x2.metric("Fixed model", ind["fixed"]["decision"]); x2.caption("same person, same data")
            st.markdown("**Change only her gender and nothing else**")
            y1, y2 = st.columns(2)
            y1.metric("Baseline, as a man", ind["baseline_if_male"]["decision"]); y1.caption(f"score {ind['baseline_if_male']['score']:.2f}")
            y2.metric("Fixed, as a man", ind["fixed_if_male"]["decision"])
            y2.caption("decision unchanged" if ind["fixed_if_male"]["decision"] == ind["fixed"]["decision"] else "decision changes")
            if ind["baseline_if_male"]["decision"] == "APPROVED":
                st.error(
                    f"The baseline rejects her at {ind['baseline']['score']:.2f} but would approve the identical applicant as a man "
                    f"({ind['baseline_if_male']['score']:.2f}). After the fix she is approved."
                )
            if ind["fixed_if_male"]["decision"] != ind["fixed"]["decision"]:
                st.info("The fixed model uses a separate cut-off for each group by design, so it does not treat the two versions identically. "
                        "This trade-off is listed under Limits.")
        st.caption(
            f"Illustration only. It comes from a typical split (one whose numbers are close to the 20-split average). Of the qualified women the baseline "
            f"rejects and the fixed model approves, this is the one whose score rises most when only gender changes "
            f"({ind['n_candidates']} such women in this split)."
        )
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    st.subheader("It is not just one person")
    a, b, c3 = st.columns(3)
    n_good = pop["n_good_women_test"]
    a.metric("Qualified women rejected, baseline", f"{pop['baseline_rejected']:.1f} of {n_good:.0f}")
    b.metric("Qualified women rejected, fixed", f"{pop['fixed_rejected']:.1f} of {n_good:.0f}")
    c3.metric("Decisions that flip if only gender changes", f"{pct(pop['baseline_flip_rate'])} to {pct(pop['fixed_flip_rate'])}")
    st.caption(
        f"Average of {R['multi_seed']['n_splits']} random splits (about {n_good:.0f} qualified women in each test set). "
        "Gender-flip test: take every woman in the test set, change only her gender, and count how often the decision changes."
    )

# --------------------------------------------------------------- SUBGROUPS
with t7:
    st.subheader("Does the gap look the same in every kind of applicant?")
    st.markdown(
        "The fix equalises approval rates between men and women **overall**. This tab checks the gap inside smaller groups. "
        "Each bar is the approval gap (men minus women) for applicants of one type, pooled over the random splits."
    )
    LABEL = {"Property_Area": "Property area", "Married": "Married", "Education": "Education", "Self_Employed": "Self-employed"}
    rows = [r for r in R["multi_seed"]["subgroups"] if r["baseline_gap"] is not None]
    rows.sort(key=lambda r: (list(LABEL).index(r["column"]), r["value"]))
    names = [f"{LABEL[r['column']]}: {r['value']}  ({r['n_women']} women)" for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(rows) + 1.2))
    for off, key, col, lab in ((0.27, "baseline_gap", BASE_C, "Baseline"), (0.0, "fixed_gap", FIXED_C, "Fixed (threshold)"), (-0.27, "eg_gap", "#7a4fc1", "In-training")):
        ax.barh(y + off, [100 * (r[key] or 0) for r in rows], 0.26, color=col, label=lab)
    ax.axvline(0, color=GRID, lw=1)
    ax.set_yticks(y, names, fontsize=9, color=INK)
    ax.set_xlabel("approval gap, men minus women (percentage points)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(colors=MUTED, length=0); ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); st.pyplot(fig)
    big = [r for r in rows if r["n_women"] >= 20 and r["fixed_gap"] is not None]
    worst = max(big, key=lambda r: abs(r["fixed_gap"])) if big else None
    txt = ("The baseline gap is large in most groups. " if sum(r["baseline_gap"] > .2 for r in rows) > len(rows) / 2 else "")
    if worst:
        txt += (f"After the fix, most groups are close to equal, but <b>{LABEL[worst['column']].lower()} = {worst['value']}</b> still shows a gap of "
                f"{pts(abs(worst['fixed_gap']))}. Equal rates overall do not guarantee equal rates inside every subgroup.")
    st.markdown(f"<div class='note'>{txt}</div>", unsafe_allow_html=True)
    st.caption("Some groups have very few women (the count is in each label), so treat individual bars as a pointer for where to look next, not as a finding. "
               "Because the bias in this demo was injected evenly, differences between groups here come from the model and the small samples, not from separate biases we added.")

# -------------------------------------------------------------- 4. TRY IT
with t4:
    st.subheader("Try it yourself")
    st.markdown("Change the applicant's details. The same person is always scored twice, once as a woman and once as a man, by both models.")
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
                     "Approval score (baseline)": f"{M.score(p)[0]:.2f}",
                     "Baseline decision": "Approved" if M.baseline_decision(p)[0] else "Rejected",
                     "Fixed decision": "Approved" if M.fixed_decision(p)[0] else "Rejected"})
    st.table(pd.DataFrame(rows).set_index("Same applicant…"))
    st.caption("Starting values are the most common ones in the data. Demo only. Not a real credit decision.")

# ------------------------------------------------------------ 5. IMPACT
with t5:
    pop = R["multi_seed"]["population"]
    st.subheader("Who this helps")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("**Banks**")
        st.markdown(
            "Test a lending model for hidden group bias before it goes live, and compare fixes on the same data. "
            f"In this demo the gender-flip test shows {pct(pop['baseline_flip_rate'])} of decisions for women changing before the fix "
            f"and {pct(pop['fixed_flip_rate'])} after."
        )
    with b2:
        st.markdown("**Applicants**")
        st.markdown(
            "A rejected applicant rarely learns why. Group-level tests like these make it possible to check whether "
            "a protected attribute alone is moving decisions, which makes decisions easier to audit and challenge."
        )
    with b3:
        st.markdown("**Regulators**")
        st.markdown(
            "Evidence instead of a claim. A lender can show the measured gap before and after a fix, "
            "with the method and its limits written down, rather than stating that its model is fair."
        )
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    st.subheader("Audit report")
    st.markdown("The steps above (model, test, measure, fix, re-test, limits) written up as a plain-text report you can hand to someone.")
    st.download_button("Download audit report (.txt)", audit_report(), file_name="fairloan_audit_report.txt", mime="text/plain")
    with st.expander("Preview the report"):
        st.code(audit_report(), language=None)
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    st.subheader("From prototype to a tool a lender could use")
    st.markdown(
        """
1. **Validate the data.** This prototype uses a public Indian loan dataset. The next step is a lender's own data, with checks on where it came from and how the protected attribute was recorded.
2. **More groups.** Test beyond gender: rural against urban, marital status, and combinations of groups.
3. **Fairness without the group at decision time.** The threshold fix needs gender when it scores someone. We already tested an in-training method that does not, and it closes most of the gap here. The next step is testing it on larger and messier data.
4. **A proper audit workflow.** A bank uploads a model, the tool tests it, applies a fix, tests again, and produces the report above.
"""
    )

# ---------------------------------------------------------------- 5. LIMITS
with t6:
    c = R["config"]
    st.subheader("What this does and does not show")
    st.markdown(f"""
- **The bias is injected, on purpose.** In the raw data women were approved {pct(c['raw_approval_women'])} of the time and men {pct(c['raw_approval_men'])}, almost the same. So the unfairness this demo shows is the one we put in. We can only *prove* a model copied unfairness if we know the fair answer, so we corrupted training labels ourselves and kept test labels clean. Real datasets rarely let you verify this; there you can only measure outcome gaps.
- **Small data.** {c['n_train'] + c['n_test']} applicants, of whom {c['n_women_total']} are women; about {c['n_women_test']} women in each test set. That is why headline numbers average 20 random splits, and single-split numbers move around (baseline qualified-approval gap: ±{pts(SD[B]['equal_opportunity_gap'])}).
- **Public Indian housing-finance data, not bureau data.** Real CIBIL data is private. This is a public loan-eligibility dataset. Its label is the lender's recorded decision (approved or not), not a confirmed repayment outcome. Check its licence before publishing.
- **Fairness metrics conflict.** We equalise approval rates (demographic parity). Equal error rates (equalized odds) can't be fully satisfied at the same time; our fixed model still has an error-rate gap (equalized-odds difference {MS[F]['eo_diff']:.2f}).
- **Uses gender at decision time.** ThresholdOptimizer needs the group label when scoring, so it applies a different cut-off to each group; that is restricted in some jurisdictions. Fairlearn's in-training `ExponentiatedGradient` avoids that. We tested it (see the Fix tab); it works here, but on about 600 rows its result moves around from split to split.
- **Gender is a stand-in.** It is recorded only as Male or Female, and the {c['n_dropped_no_gender']} applicants with no recorded gender were dropped rather than guessed. Marital status, education, urban or rural area and combinations of these are worth auditing too.
- **Deleting the gender column** already closes most of the gap here, because the other columns barely hint at gender. In data with strong hints such as pin code it would help much less; we did not test that.
""")
