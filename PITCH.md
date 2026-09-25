# FairLoan: 3-minute pitch (Indian loan data)

Numbers are the average of 20 random splits on the Dream Housing Finance data (601 applicants), measured against true labels, unless stated. Confirm rupee units on the dataset page before saying them aloud.

**0:00 Hook (20 s)**
"Banks and lenders in India now use AI to help decide who gets a loan. That AI learns from old decisions. If those decisions were unfair, it learns the unfairness, and nobody ever told it to. The RBI's own FREE-AI report found that only 10% of the lenders it surveyed had bias-mitigation protocols."

**0:20 Catch it (50 s)** *(Tab ①)*
"We took real Indian loan data and recreated the problem on purpose: in the training history, 30% of women who actually repaid were recorded as defaults. Then we trained a completely normal model. Among applicants who would have repaid, it approves 98% of men but only 70% of women, a 27-point gap. Overall it approves 86% of men and 57% of women. The test labels are clean, so this isn't a guess: we know who deserved the loan."

**1:10 Fix it (50 s)** *(Tab ②)*
"We use Fairlearn's ThresholdOptimizer. The model stays the same; it learns one approval cut-off per group so both are approved at equal rates. The approval gap falls from 29 points to 2, and accuracy against the true labels doesn't drop: 0.781 to 0.793. The chart shows it holds as the historical bias grows from 0 to 40%."

**2:00 Prove it (50 s)** *(Tab ③)*
"Meet an unmarried graduate, self-employed, with a good credit history, asking for a 150-thousand-rupee loan over 30 years. She would repay. The baseline rejects her with a score of 0.46. Change only her gender to male and the same model scores 0.71 and approves. After the fix she is approved. Across all splits, the baseline rejects about 7 of every 23 qualified women; after the fix, about 1. And changing only gender flips 27% of the baseline's decisions for women, versus 2% after the fix."

**2:50 Close (10 s)**
"Bias in, bias out. We can measure it, fix it, and show the person it hurt."

---
## Judge Q&A

**Why not just delete the gender column?** Honestly, on this dataset it works about as well: the gap falls from 29 to 3 points, and accuracy is even slightly higher (0.802 versus 0.793). The other columns only weakly hint at gender (gender can be guessed from them with AUC 0.74, where 0.5 is a coin flip), so there is little left for the bias to hide in. What Fairlearn adds is a measurable guarantee of equal approval rates that does not depend on how strong those hints are, and it works when you can't or shouldn't drop a column. We did not test data with strong hints such as pin code. [Say this plainly.]

**Isn't the bias fake?** Injected on purpose. In the raw data women were approved 67% of the time and men 69%, almost the same. Real data only lets you show outcome gaps, not their cause. Injecting bias while keeping test labels clean is the only way to *prove* the model copied it and that the fix removed it.

**What's the accuracy cost?** Measured against true labels: none (0.781 to 0.793). A model trained on fair labels, which real lenders never have, scores 0.802.

**Is this real CIBIL data?** No. Bureau data is private. This is a public loan-eligibility dataset in an Indian housing-finance setting.

**Is it legal to use gender in the decision?** ThresholdOptimizer needs the group at decision time and applies different cut-offs per group, which some jurisdictions restrict. Fairlearn's ExponentiatedGradient trains fairness in without needing it at prediction time. We flag this as a limitation.

**Only about 600 rows?** Yes, and only about 34 women in each test set, so single-split numbers swing a lot (the baseline's qualified-approval gap is 27 ± 17 points across splits). That is why we average 20 splits and show the spread.

**Other fairness metrics?** They can't all be satisfied together. We equalise approval rates; a gap in error rates remains (equalized-odds difference about 0.11).
