# Temporal model evaluation: Maharashtra, test filing year 2014

## Evaluation design

- Training cohorts: 2010–2012; model training used a random 40% sample of eligible cases from those years.
- Validation cohort: 2013; threshold tuning used a random 20% sample.
- Held-out test cohort: 2014; metrics use a random 20% sample, with n=222,546.
- Categories and act/section encoders were fitted on training years only. Thresholds for learned models were selected by validation-year F1 and then frozen for test scoring.

The results estimate performance on later filing-year cases with observed labels in this state. They do not establish performance across India or future source releases.

## Outcome definition and censoring

A case is positive if its recorded decision is later than the third calendar anniversary of filing. A case with no recorded decision is also labeled positive only when the year-specific disposition lookup marks disposition as missing and a plausible `date_last_list` is after that anniversary. The listing-date plausibility ceiling is 2021-12-31, used to reject implausible far-future dates; it is not treated as a source snapshot cutoff. Cases without sufficient recorded follow-up remain censored and are excluded.

## Test results

The best ROC-AUC is **logistic_regression_balanced** (ROC-AUC 0.773, PR-AUC 0.564, F1 0.623, recall 0.827, precision 0.499). Its threshold (0.464) was selected on validation data.
The highest PR-AUC is **historical_court_case_type_rate** at 0.571. k-NN was trained on 20,000 and evaluated on 5,000 sampled cases because of the high-dimensional sparse feature space; its metrics are not directly comparable with full test-sample metrics.

| Model | Test n | ROC-AUC | PR-AUC | F1 | Recall | Precision | Brier | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression_balanced | 222,546 | 0.773 | 0.564 | 0.623 | 0.827 | 0.499 | 0.213 | 0.464 |
| multinomial_naive_bayes | 222,546 | 0.757 | 0.548 | 0.611 | 0.820 | 0.487 | 0.227 | 0.370 |
| knn_sampled | 5,000 | 0.750 | 0.548 | 0.607 | 0.809 | 0.486 | 0.191 | 0.267 |
| historical_court_case_type_rate | 222,546 | 0.749 | 0.571 | 0.597 | 0.783 | 0.482 | 0.186 | 0.315 |
| decision_tree_balanced | 222,546 | 0.717 | 0.488 | 0.603 | 0.800 | 0.484 | 0.209 | 0.525 |
| prevalence_baseline | 222,546 | 0.500 | 0.332 | 0.000 | 0.000 | 0.000 | 0.222 | 0.500 |

Probability calibration is reported with Brier scores and a reliability plot. Raw model scores should not be described as literal case-level probabilities unless calibration is adequate.

## Label sensitivity
- Listing-date ceiling 2019-12-31: 1,101,243 eligible; delay rate 32.6%.
- Listing-date ceiling 2020-12-31: 1,112,734 eligible; delay rate 33.3%.
- Listing-date ceiling 2021-12-31: 1,112,735 eligible; delay rate 33.3%.
- Listing-date ceiling 2025-12-31: 1,112,736 eligible; delay rate 33.3%.

## Uncertainty and limits

A case-level bootstrap 95% interval for the best model's ROC-AUC is 0.771–0.775 (100 replicates). This interval does not account for clustering by court.
The listing-date rule and 2021 plausibility ceiling affect which pending cases are labelable. The DDL portal does not provide an exact extraction cutoff, and date fields contain malformed far-future values. Results therefore need a sensitivity review and should not be framed as a definitive all-case risk estimate.

The judge-relationship analysis has been removed from the active project scope at the user's direction. No judge-related results are included here.

## Reproducibility files

Code is under `codebase/`. Cohort, feature, prediction, model, metric, sensitivity, and chart artifacts are under `result/`. Case-type, court, and act/section descriptive tables summarize the held-out test filing year.
