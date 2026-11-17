# Data audit and temporal cohort

## Source and local layout

The workspace contains Development Data Lab (DDL) Indian Judicial Data CSVs under `datasetddl/csv/`. Annual case tables use `ddl_case_id` as the case key. Act/section records are multi-row per case, and lookup tables provide readable case-type, disposition, court, act, and section labels. The [DDL Judicial Data portal](https://www.devdatalab.org/judicial-data) describes the public data release and its terms; consult it for current attribution and sharing requirements.

The active analysis uses Maharashtra (state code `01`), filing years 2010–2014. Judge identity, gender, religion, and party identity analyses are out of scope because the required linkage and attributes are not suitable for the stated analysis; the earlier judge task was cancelled.

## Target definition and unresolved cases

For valid filing and decision dates, a case is delayed when the decision date is later than the third calendar anniversary of filing. When the decision is missing and its year-specific disposition lookup marks the disposition as missing, a plausible last-list date strictly after the anniversary establishes that the case remained without recorded disposition after three years. Other unresolved cases and invalid date records are censored/excluded. A `2021-12-31` ceiling rejects implausible far-future listing dates; it is a data-quality rule, not a verified extraction cutoff.

The 2018 Maharashtra date audit found 1,374,670 state rows, 761,016 missing/sentinel decisions, 613,654 valid recorded decisions, 90,845 decisions after 2018-12-31, and a latest valid decision of 2020-09-15. Listing fields included far-future values (last-list through 2108 and next-list through 2219). These dates do not reveal a source snapshot cutoff. Accordingly, unresolved cases are labeled only under the conservative rule above, and sensitivity to the listing-date plausibility ceiling is reported.

## Cohort flow

| Filing year | Labelable cases | Delayed cases |
|---:|---:|---:|
| 2010 | 844,505 | 299,600 |
| 2011 | 863,394 | 293,271 |
| 2012 | 961,929 | 329,497 |
| 2013 | 1,036,158 | 344,545 |
| 2014 | 1,112,735 | 370,743 |

The 2014 source file has 1,114,737 Maharashtra rows; no duplicate case IDs were removed. The remainder after applying the outcome rules is censored or invalid. See `result/tables/sample_flow_state01_year*.csv` for the detailed yearly flow.

For 2014, changing the last-list plausibility ceiling to 2019, 2020, 2021, or 2025 yields 1,101,243, 1,112,734, 1,112,735, and 1,112,736 eligible cases, respectively; the corresponding delayed prevalence is 32.6%, 33.3%, 33.3%, and 33.3%.

## Temporal evaluation

Training uses a random 40% sample of eligible 2010–2012 cases; validation uses 20% of 2013 for threshold selection; test metrics use a random 20% of 2014 (222,546 cases). Encoders are fit on training years only. k-NN uses smaller train/validation/test samples due to sparse high-dimensional features and is not directly comparable to full test-sample metrics.

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| Balanced logistic regression | 0.773 | 0.564 | 0.623 | 0.827 | 0.499 |
| Weighted multinomial Naive Bayes | 0.757 | 0.548 | 0.611 | 0.820 | 0.487 |
| Sampled k-NN | 0.750 | 0.548 | 0.607 | 0.809 | 0.486 |
| Historical court/case-type rate | 0.749 | 0.571 | 0.597 | 0.783 | 0.482 |
| Balanced decision tree | 0.718 | 0.488 | 0.603 | 0.800 | 0.484 |

A case-level bootstrap (100 replicates) gives a 95% ROC-AUC interval of 0.771–0.775 for logistic regression. It does not account for court clustering. Full metrics, thresholds, Brier scores, and intervals are in `result/tables/` and the generated conclusion.

## Descriptive patterns

The 2014 test-year tables and charts summarize observed delay rates by case type, court, and act/section. Act/section groups overlap when a case has multiple pairs. These are descriptive, unadjusted comparisons and support association statements only; they do not isolate independent effects or establish causes. No judge-linked results are part of the active analysis.

## Reproducibility

Run `bash codebase/run_pipeline.sh` from the project root. The pipeline writes cohorts/features, metrics, saved models, pattern tables/charts, and the generated conclusion into `result/` and `conclusion/`. Module descriptions are in `codebase/README.md`.
