---
title: FairLoan
emoji: ⚖️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# ⚖️ FairLoan — when the training data has a bias

Catch it → Fix it → Prove it. A loan-approval model learns an unfairness that was hiding in its historical data; we measure it, fix it with Fairlearn, and show one real person whose decision flips.

## Run it (3 commands)
```bash
pip install -r requirements.txt
python train.py          # ~20 s: trains models, writes artifacts/
streamlit run app.py     # dashboard at http://localhost:8501
```
`artifacts/` is already committed, so `streamlit run app.py` works immediately.

## What's in here
| File | Purpose |
|---|---|
| `pipeline.py` | All the ML: bias injection, baseline, Fairlearn fix, metrics, the individual case |
| `train.py` | Runs the pipeline, saves `artifacts/results.json` + `models.joblib` |
| `app.py` | Streamlit dashboard (5 tabs) |
| `FairLoan.ipynb` | Same story as a notebook for Colab / Kaggle |
| `data/india_credit.csv` | Dream Housing Finance loan data (India), 614 rows, 601 used. See `DATA_CARD.md` |
| `checks.py` | Sanity checks: run after any change (`python checks.py`) |
| `PITCH.md` | 3-minute script + judge Q&A |

## How it works
1. **Biased history.** Only the *training* labels are corrupted: 30% of women who repaid are recorded as defaults. Test labels stay clean, so every metric is measured against the truth.
2. **Catch.** Logistic regression on the biased labels. Fairlearn `MetricFrame` + `demographic_parity_difference` + `equalized_odds_difference`. "Equally qualified" = approval rate among applicants who would repay (equal-opportunity gap).
3. **Fix.** `ThresholdOptimizer(constraints="demographic_parity")` on the same model: one cut-off per group.
4. **Prove.** Same metrics after the fix, 20-split average; a rule-selected woman who truly repays, is rejected by the baseline, approved after the fix; and a gender-flip counterfactual over all women.

## Headline numbers (mean of 20 random splits, measured on clean labels)
| | Baseline | Delete gender column | Fixed (Fairlearn) |
|---|---|---|---|
| Approval gap, men − women | 29 pts | 3 pts | 2 pts |
| Gap among applicants who would repay | 27 pts | 0 pts | 0 pts |
| Accuracy | 0.781 | 0.802 | 0.793 |

The spread across splits is large (the baseline's qualified gap is 27 ± 17 pts) because each test set has only about 34 women. **Deleting the gender column works about as well as the Fairlearn fix on this data**, because the other columns barely hint at gender. The dashboard says so on the Fix tab and the Honest limits tab. Average over 20 splits: 6.8 of 23 qualified women are rejected by the baseline versus 1.1 after the fix; changing only gender flips 27% of the baseline's decisions for women versus 2%.

## Deploy (public link for judges)
**Option A: Streamlit Community Cloud (simplest).** Push this folder to GitHub → share.streamlit.io → New app → pick the repo, main file `app.py`.

**Option B: Hugging Face Spaces.** New Space → SDK *Docker* → upload these files (the README front-matter above is already set up, `Dockerfile` listens on 7860). *Not tested from the build environment; if the build fails, check the Space's build log.*

Always keep `streamlit run app.py` working locally on a laptop as a backup in case venue Wi-Fi fails.
