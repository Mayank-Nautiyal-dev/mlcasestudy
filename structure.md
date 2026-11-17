# Project scope, status, and research plan

> **Working article title:** *New Evidence on Judicial Delay*  
> **SDG:** 16 — Peace, Justice and Strong Institutions  
> **Article type:** original research article  
> **Current status:** a Maharashtra temporal evaluation has been built and run. The code, metrics, charts, and data audit are documented in `README.md`, `docs/data_audit.md`, `codebase/README.md`, `result/`, and `conclusion/`. The research paper has not been drafted.

## Research questions

1. At filing, can case type, act/section, court, and filing year predict whether a case will take more than three years to reach recorded disposal?
2. How do observed delay rates vary across case types, act/section pairs, and courts in the selected cohort?

The judge/party identity relationship task has been cancelled. No judge gender, religion, caste, or judge-assignment analysis is part of the active scope. The supplied judicial-bias article remains a background reference only; its identity and causal methods are not reproduced here.

Prediction and descriptive patterns answer different questions. Model scores and category rates do not show that a feature causes delay.

## Data and cohort

The source is the Development Data Lab Indian Judicial Data, derived from e-Courts, organized locally as CSVs in `datasetddl/csv/`. The first geography is Maharashtra (state code `01`). The temporal experiment uses filing cohorts 2010–2014.

The case files contain `ddl_case_id`, filing year, state/district/court codes, case type, filing/decision/last-list dates, disposition fields, and party/advocate gender indicators. The separate act/section file has multiple rows per case and is aggregated to a set of act/section code tokens. Year-specific key files supply readable case-type and court labels.

### Three-year target

- If a valid decision date exists, a case is positive when the decision occurs later than the third calendar anniversary of filing; otherwise it is negative.
- If the decision date is missing and the year-specific disposition lookup marks the disposition variable as missing, the case is positive only when a plausible last-list date is later than the three-year anniversary.
- For pending-case labeling, `2021-12-31` is used as a date-quality ceiling to reject implausible far-future listing dates. This is not assumed to be the source snapshot cutoff.
- Cases without a valid decision or enough valid listing-date follow-up are censored and excluded. Decision-before-filing records and other inconsistent dates are also excluded.

This rule uses observable case history to identify cases already beyond the three-year mark while avoiding an unsupported claim about the overall extraction date. The sensitivity table compares alternative listing-date ceilings.

## Temporal evaluation

- **Training:** filing years 2010–2012; a random 40% sample of eligible cases is used for model fitting.
- **Validation:** filing year 2013; a random 20% sample selects decision thresholds.
- **Test:** filing year 2014; a random 20% sample is scored after model selection.
- All category encoders are fitted on training years only. The test year is not used for threshold selection.

The models are prevalence baseline, historical court/case-type rate baseline, class-balanced logistic regression, class-balanced decision tree, Naive Bayes with class-balanced sample weights, and k-NN on a smaller stratified sample (20,000 training, 5,000 validation, 5,000 test cases). Because k-NN uses a different test sample size, its scores are not directly comparable with the full temporal test-sample metrics.

The primary metrics are ROC-AUC, PR-AUC, F1, recall, precision, accuracy, and Brier score. Thresholds for learned models are tuned on validation F1. Case-level bootstrap intervals are provided for the full-sample models; they do not account for court-level clustering.

## Current results

The test sample contains 222,546 cases and has a 33.2% delay rate. Balanced logistic regression leads ROC-AUC (0.773), F1 (0.623), and recall (0.827). The historical court/type rate baseline has the highest PR-AUC (0.571). The logistic ROC-AUC case-level bootstrap interval is 0.771–0.775 (100 replicates).

The listing-date ceiling sensitivity ranges from 32.6% to 33.3% delay prevalence in 2014. Model calibration still needs care: the Brier score and reliability plot show that raw outputs should not automatically be interpreted as calibrated case-level probabilities.

## Codebase and outputs

- `codebase/`: audit, cohort construction, multi-year feature building, temporal training, k-NN, threshold tuning, label sensitivity, bootstrap intervals, pattern analysis, and pipeline scripts.
- `result/data/`: yearly and combined cohorts, features, held-out predictions, and sampled k-NN predictions.
- `result/models/`: fitted temporal classifiers and preprocessing.
- `result/tables/`: sample flow, date coverage, label sensitivity, model metrics, bootstrap intervals, and case/court/act-section summaries.
- `result/figures/`: temporal ROC, PR, calibration, confusion matrix, and delay-pattern charts.
- `conclusion/`: generated temporal evaluation summary.
- `docs/data_audit.md`: verified field inventory, cohort counts, label construction, and limitations.

Run from the project root with `bash codebase/run_pipeline.sh`. The complete module list is in `codebase/README.md`.

## Limitations and remaining work

- Results cover one state and test one later filing-year cohort; they are not national estimates.
- The exact source extraction cutoff is not documented. Pending cases without sufficient observed follow-up remain censored.
- Date fields include sentinels and implausible far-future listing dates; the listing-date ceiling needs robustness review.
- The model-training, validation, and test cohorts are random samples of their eligible filing-year rows for computational scale.
- The judge/party identity analysis is out of scope by user direction.
- The research paper remains to be drafted after final review of outcome robustness, calibration, and model explanations.

## Data terms and reference paper

The DDL portal describes the public release as 2010–2018 and lists CC BY-NC-SA 4.0 terms for non-commercial use. Check the current [DDL Judicial Data portal](https://www.devdatalab.org/judicial-data) for attribution and sharing requirements. It says replication of the judicial-bias paper uses a separate dataset; we use the supplied paper only as background context.
