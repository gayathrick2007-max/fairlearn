import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s)); code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# FairLoan: catch it, fix it, prove it
**Run in Google Colab or Kaggle.** Upload this whole folder (or `git clone` your repo), then Run all. Data: `data/india_credit.csv` (see DATA_CARD.md).
The heavy lifting lives in `pipeline.py` so the notebook, the dashboard and the pitch all show the same numbers.""")
code("!pip -q install fairlearn  # Colab already has pandas / scikit-learn / matplotlib\n# If you cloned a repo: %cd your-repo-name")
code("""import numpy as np, pandas as pd, matplotlib.pyplot as plt
import pipeline as pl
df = pl.load_data()
print(df.shape, '| share of women:', round(df.female.mean(), 2))
print('Truly good loans - men:', round(df[df.female==0].credit_risk.mean(), 2), ' women:', round(df[df.female==1].credit_risk.mean(), 2))""")

md("""## 0. Recreate the problem: a biased history
We corrupt **only the training labels**: 30% of women who really repaid get recorded as defaults. Test labels stay clean, so we know the truth.""")
code("""m = pl.FairLoanModels(seed=pl.SHOWCASE_SEED, bias=pl.DEFAULT_BIAS).fit()
flipped = (m.ytr_biased != m.ytr_true).sum()
print(f'{flipped} training labels flipped')
print('Recorded "good" rate in training - men:', m.ytr_biased[m.atr==0].mean().round(2), ' women:', m.ytr_biased[m.atr==1].mean().round(2))
print('TRUE "good" rate in training     - men:', m.ytr_true[m.atr==0].mean().round(2),  ' women:', m.ytr_true[m.atr==1].mean().round(2))""")

md("## 1. CATCH it\nAn ordinary logistic regression trained on that history. Everything is measured against the *true* test labels.")
code("""res = pl.evaluate_all(m)
b = res['Baseline']
print(pd.DataFrame(b['by_group']).T.round(3))
print()
print('Approval gap (men - women):        ', round(b['approval_gap'], 3))
print('Equal-opportunity gap (qualified): ', round(b['equal_opportunity_gap'], 3))
print('Fairlearn demographic parity diff: ', round(b['demographic_parity_difference'], 3))
print('Fairlearn equalized odds diff:     ', round(b['equalized_odds_difference'], 3))""")
code("""# Single splits are noisy with only 300 test rows - average 20 random splits
ms = pl.multi_seed(0.30, range(20))
pd.DataFrame(ms['mean']).T[['accuracy','male_approval','female_approval','male_tpr','female_tpr','approval_gap','equal_opportunity_gap']].round(3)""")

md("## 2. FIX it\nFairlearn `ThresholdOptimizer` (demographic parity) on top of the *same* trained model.")
code("""print('Learned cut-offs on repay-probability:', m.thresholds())
f = res['Fixed (Fairlearn)']
print(pd.DataFrame(f['by_group']).T.round(3))
print('Accuracy  baseline -> fixed:', round(b['accuracy'],3), '->', round(f['accuracy'],3))
print('Approval gap  ->', round(f['approval_gap'],3), '| qualified gap ->', round(f['equal_opportunity_gap'],3))""")
code("""names = ['Baseline', 'Baseline without gender column', 'Fixed (Fairlearn)']
x = np.arange(3); w = .38
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for a, key, ttl in zip(ax, ['approval_rate', 'tpr'], ['Approval rate, everyone', 'Approval rate, equally qualified']):
    a.bar(x - w/2, [ms['mean'][n]['male_approval' if key=='approval_rate' else 'male_tpr'] for n in names], w, color='#2a78d6', label='Men')
    a.bar(x + w/2, [ms['mean'][n]['female_approval' if key=='approval_rate' else 'female_tpr'] for n in names], w, color='#eb6834', label='Women')
    a.set_xticks(x, ['Baseline', 'Drop gender col', 'Fixed'], fontsize=9); a.set_title(ttl, loc='left'); a.set_ylim(0, 1); a.legend(frameon=False)
plt.tight_layout(); plt.show()""")

md("## 3. PROVE it: one real person\nA woman who truly repays, rejected by the baseline, approved after the fix; plus the gender-flip counterfactual.")
code("""ind = pl.find_individual(m)
print(pd.Series(ind['profile']).to_string())
print()
print('Truth:            ', ind['true_outcome'])
print('Baseline:         ', ind['baseline']['decision'], f"(score {ind['baseline']['score']:.2f}, needs 0.50)")
print('Baseline if MALE: ', ind['baseline_if_male']['decision'], f"(score {ind['baseline_if_male']['score']:.2f})")
print('Fixed:            ', ind['fixed']['decision'])
print('Fixed if MALE:    ', ind['fixed_if_male']['decision'])""")
code("""print(pl.wrongly_rejected_counts(m))
print(pl.counterfactual_flip_rates(m))
print(pl.proxy_leakage())""")
md("""## Limits worth saying out loud
* The bias is **injected** so we can prove it. Real data only lets you measure gaps.
* About 600 applicants and roughly 34 women in each test set, so we average many splits.
* Demographic parity and equalized odds can't both be perfect. Our fixed model still has an error-rate gap.
* `ThresholdOptimizer` needs gender at decision time; `ExponentiatedGradient` avoids that.
* Deleting the gender column closes most of the gap on this dataset (weak proxies), so the Fairlearn fix is a tie with it here.""")
nb["cells"] = C
nbf.write(nb, "FairLoan.ipynb")
