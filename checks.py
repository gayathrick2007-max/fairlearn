"""Sanity checks. Run after ANY change to the data or pipeline:  python checks.py"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pipeline as pl

ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name + (f"  ({detail})" if detail else ""))

df = pl.load_data()
check("gender column has both groups", set(df.female.unique()) == {0, 1}, f"{int(df.female.sum())} women of {len(df)}")
check("outcome is 0/1", set(df.credit_risk.unique()) == {0, 1}, f"approval rate {df.credit_risk.mean():.2f}")
check("no blanks left", int(df.isna().sum().sum()) == 0)

m = pl.FairLoanModels(seed=3, bias=0.30).fit()
_, _, ytr, yte, atr, _ = pl.split_data(3)
check("TEST labels are the untouched originals", np.array_equal(m.yte, yte))
flipped = m.ytr_biased != m.ytr_true
check("only women were flipped", bool((m.atr[flipped] == 1).all()))
check("only truly-good women were flipped, good -> default", bool((m.ytr_true[flipped] == 1).all() and (m.ytr_biased[flipped] == 0).all()))
shares = []
for sd in range(20):   # one split has only ~50 qualified women, so judge the AVERAGE share, not a single split
    _, _, y_, _, a_, _ = pl.split_data(sd)
    shares.append((pl.inject_bias(y_, a_, 0.30, sd) != y_).sum() / ((a_ == 1) & (y_ == 1)).sum())
check("about 30% of truly-good women flipped in training (average of 20 splits)", abs(np.mean(shares) - 0.30) < 0.05,
      f"mean {np.mean(shares):.0%}, single splits range {min(shares):.0%} to {max(shares):.0%}")

gaps = {}
for p in (0.0, 0.30):
    gaps[p] = np.mean([pl.evaluate_all(pl.FairLoanModels(seed=s, bias=p).fit())["Baseline"]["approval_gap"] for s in range(5)])
check("baseline gap is far larger with injected bias than without", gaps[0.30] > gaps[0.0] + 0.10, f"{gaps[0.0]:.2f} -> {gaps[0.30]:.2f}")

r = pl.evaluate_all(m)
check("fixed model shrinks the approval gap", abs(r["Fixed (Fairlearn)"]["approval_gap"]) < abs(r["Baseline"]["approval_gap"]) / 2)
ind = pl.find_individual(m)
check("a demo applicant exists", bool(ind), f"{ind.get('n_candidates')} candidates" if ind else "")
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
