# Data card

**Dataset.** Dream Housing Finance loan-eligibility data (train file), a public Indian-style loan dataset. The team downloaded it from Kaggle. **To fill in before submission:** exact Kaggle page URL, licence, and the column definitions from that page. This is *not* real CIBIL bureau data, which is private.

**File used:** `data/india_credit.csv` (made from the train file: `Loan_Status` Y/N became `credit_risk` 1/0).

| Item | Value |
|---|---|
| Rows in the file | 614 applicants |
| Rows used | 601 (13 applicants with no recorded gender were dropped, not guessed) |
| Gender | 489 male, 112 female, 13 blank in the file |
| Outcome (`credit_risk`) | 422 approved (1), 192 rejected (0) |
| Approval rate, raw data | men 69.3%, women 67.0% |
| Columns used as inputs | Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area (+ gender, which the baseline can see) |
| Dropped | Loan_ID (an identifier) |

**Blanks in other columns** (Married 3, Dependents 15, Self_Employed 32, LoanAmount 22, Loan_Amount_Term 14, Credit_History 50 in the file) are filled with the median for numbers and the most common value for text. This is simple, not perfect.

**Units.** Incomes appear to be rupees per month and `LoanAmount` thousands of rupees. Confirm on the source page before quoting rupee figures.

**Known limits.**
- Only about 34 women in each test split, so single-split numbers are noisy; headline numbers average 20 random splits.
- In the raw data women and men are approved at almost the same rate. The unfairness shown in the demo is the bias we inject into training labels on purpose; test labels stay clean.
- Gender is recorded as Male or Female only.
