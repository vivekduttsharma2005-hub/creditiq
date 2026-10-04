# CreditIQ: calibrated credit default scoring

> **Educational model, not a lending tool.** Not validated for real credit decisions. No claim of regulatory compliance.

## 1. Problem spec (Day 1)
- **Target:** `default` = 1 if loan_status is *Charged Off*, 0 if *Fully Paid*.
- **Dropped:** loans that are *Current*, *Late* or *In Grace Period*. Their outcome is not known yet, so they cannot be labelled.
- **Prediction moment:** the day the loan is issued. Only information the lender knew then is used.
- **Success:** beat the logistic baseline on PR-AUC on a *later* time period, with calibrated probabilities and a cost-based threshold.

## 2. Keep / drop list (leakage control)
| Column | Decision | Reason |
|---|---|---|
| loan_amnt, term, installment, purpose | Keep | Set at application |
| annual_inc, dti, emp_length, home_ownership, verification_status | Keep | Reported at application |
| fico_range_low / high | Keep | Score at application |
| delinq_2yrs, revol_util, open_acc, total_acc, pub_rec, revol_bal, earliest_cr_line | Keep | Credit file at application |
| int_rate, grade | Optional | Set by the lender's own model; train with and without, report both |
| total_pymnt, total_rec_prncp, total_rec_int | Drop | Payments received after issue |
| recoveries, collection_recovery_fee | Drop | Only exist after a default |
| last_pymnt_d, last_pymnt_amnt, next_pymnt_d, last_credit_pull_d | Drop | Recorded later |
| out_prncp | Drop | Balance after issue |
| last_fico_range_*, hardship_*, settlement_* | Drop | Recorded later |

`tests/test_leakage.py` fails if a dropped column enters the features.

## 3. Known limitations
- Only *accepted* loans are in the data. Rejected applicants are never seen (reject inference problem).
- Recent loans look safer only because many are unresolved (censoring).

## 4. Run
    pip install -r requirements.txt
    python -m src.sample_data      # FAKE practice data, or put the real file in data/accepted.csv
    python -m pytest -q