# Detailed repository guide

This guide explains the project from the raw DDL files through the trained models and result artifacts. The recommended reading order is: **scope → source data → target/cohort → features → temporal evaluation → findings → limitations**.

## 1. Project in one page

The project studies whether information available at filing can help identify Maharashtra district-court cases that will take more than three years to reach recorded disposal. It also describes how observed delay rates vary by case type, court, and act/section.

- **Source:** Development Data Lab (DDL), Indian Judicial Data, derived from e-Courts. The DDL portal describes its public data and terms: <https://www.devdatalab.org/judicial-data>.
- **Study subset:** Maharashtra (state code `01`), filing years 2010–2014.
- **ML task:** Binary classification.
- **Target:** `delay_over_3_years` (1 = over three years; 0 = within three years). Cases without enough reliable follow-up are excluded as censored.
- **Temporal split:** train on 2010–2012, tune decision thresholds on 2013, report final test metrics on 2014.
- **Predictors actually supplied to the classifiers:** `court_name`, `case_type`, `year`, and the aggregated `act_section_tokens` text field.
- **Judge analysis:** Cancelled. The codebook and raw files may contain judge-related material, but judge identity, religion, gender, and judge effects are not included in this study's analyses or model inputs.
- **Research article:** A journal-style draft is available as `judicial_delay_temporal_paper_draft.pdf`; author and course metadata still need review.

This is a state-level observational study. Prediction scores show how well a model ranks or classifies cases in the held-out 2014 sample; group delay rates are descriptive. Neither establishes that a court, case type, or law causes delay.

## 2. Repository map

| Path | What it contains | How to use it |
|---|---|---|
| `README.md` | Short project overview, headline results, setup and limits | Start here for the summary. |
| `structure.md` | Research scope, questions, methods, and roadmap | Use to see the project's intended research framing. |
| `detailedsteps.md` | This full repository walkthrough | Use to understand each stage and artifact. |
| `judicial_delay_temporal_paper_draft.pdf` | Journal-style draft with the project results | Review and complete remaining author/course declarations before submission. |
| `datasetddl/csv/` | Original DDL CSV data used by the project | Source inputs; do not edit these files. |
| `codebase/` | Python pipeline, module guide, dependencies, and run script | Reproduce the analysis. |
| `result/data/` | Filtered cohorts, joined features, predictions | Inspect case-level derived data; handle case IDs carefully. |
| `result/models/` | Serialized fitted preprocessing and model objects | Reload models with compatible Python/scikit-learn versions. |
| `result/tables/` | Cohort flow, metrics, sensitivity, uncertainty, and pattern tables | Main numeric evidence for the analysis. |
| `result/figures/` | Evaluation plots and descriptive pattern charts | Visual summaries of the result tables and model scores. |
| `conclusion/` | Generated evaluation summary | Read after the metrics and limitations. It is not the paper. |
| `docs/data_audit.md` | Detailed data definition, audit findings, and known limitations | Consult for label and source-date decisions. |

### 2.1 Raw dataset layout

The relevant local paths are:

```text
datasetddl/csv/
├── cases/cases/cases_2010.csv ... cases_2018.csv
├── acts_sections.csv
├── judges_clean.csv
└── keys/keys/
    ├── act_key.csv
    ├── section_key.csv
    ├── type_name_key.csv
    ├── disp_name_key.csv
    ├── purpose_name_key.csv
    ├── cases_court_key.csv
    ├── cases_district_key.csv
    ├── cases_state_key.csv
    └── judge_case_merge_key.csv
```

The nested `cases/cases` and `keys/keys` directories are the layout present in this workspace. Annual case tables are large, so the pipeline reads them in chunks and filters to the selected state.

## 3. What the source files contain, and what this project uses

### 3.1 Annual case records

Each `cases_YYYY.csv` contains case-level records for a filing year. The cohort builder selects these columns:

| Source field | Role in this project |
|---|---|
| `ddl_case_id` | Case key used to deduplicate and join the act/section table. Not a model predictor. |
| `year` | Filing year. Used to assign train/validation/test roles and included as a categorical predictor. |
| `state_code`, `dist_code`, `court_no` | Geography/court codes. State filters the cohort; district and court number form `court_id`; court lookup creates `court_name`, which is a model predictor. |
| `cino` | Source case identifier retained in derived cohort output; not a predictor. |
| `type_name` | Coded case type. Joined to `type_name_key.csv` to create readable `case_type`, a predictor. |
| `purpose_name` | Hearing-purpose code retained in the cohort; not a model predictor. |
| `disp_name` | Disposition code used with `disp_name_key.csv` to interpret missing disposition status and label eligible unresolved cases; not a predictor. |
| `date_of_filing` | Start date for the three-year anniversary and duration. Not supplied to the model as a raw date. |
| `date_of_decision` | End date for disposed cases and target construction. Never a model input. |
| `date_last_list` | Follow-up evidence for a subset of unresolved cases. Never a model input. |
| `judge_position` | Retained in raw cohort output but excluded from model features and judge analysis. |
| `female_defendant`, `female_petitioner`, `female_adv_def`, `female_adv_pet` | Party/advocate gender indicators retained in cohort output but not used as model features or analyzed in this study. |

The builder adds derived columns such as `delay_over_3_years`, `outcome_status`, parsed date columns, `duration_days`, and `court_id`. These are for label construction, checking, and analysis; the model matrix is restricted to the four feature fields listed above.

### 3.2 Act and section records

`acts_sections.csv` contains one or more act/section rows per case, linked by `ddl_case_id`. The feature builder selects `ddl_case_id`, `act`, and `section`, converts each distinct pair into a token such as `A4071_S1345422`, and combines tokens for each case into `act_section_tokens`. The model treats these as binary text features: a token is present or absent for a case. This avoids treating code numbers as numeric quantities.

A case can have multiple act/section tokens. Consequently, the descriptive act/section table reports overlapping groups: the same case can contribute to multiple act/section rates.

### 3.3 Lookup and auxiliary files

- `type_name_key.csv`: maps coded case types to readable `type_name_s` labels; used to make `case_type`.
- `cases_court_key.csv`: maps filing year and court codes to `court_name`; used as a categorical feature and for court summaries.
- `disp_name_key.csv`: identifies disposition labels, including a missing-disposition code; used in the target/censoring rule.
- `act_key.csv` and `section_key.csv`: support act/section labels in pattern summaries.
- District/state and purpose lookup files provide labels but are not model predictors in this analysis.
- `judges_clean.csv` and `judge_case_merge_key.csv` exist in the source bundle. Their presence does not mean judge attributes are analyzed. The judge relationship task was explicitly cancelled and is excluded from the active pipeline and claims.

## 4. Cohort and target construction

The outcome is intended to answer whether a case has taken **more than three calendar years** from filing to recorded disposal.

1. Parse filing and decision dates; treat known sentinel decision dates (for example, year `5000`) as missing.
2. Compute the third calendar anniversary of filing.
3. For a valid decision date on or before that anniversary, assign `delay_over_3_years = 0`.
4. For a valid decision date after that anniversary, assign `delay_over_3_years = 1`.
5. For a missing decision, label a case positive only if the year-specific disposition lookup says the disposition is missing and a valid `date_last_list` is after the three-year anniversary.
6. Exclude other unresolved/censored cases, invalid filing dates, and inconsistent decision-before-filing records from model evaluation.

`2021-12-31` is a plausibility ceiling for last-list dates, to reject obvious far-future data errors. It is **not** claimed to be the DDL extraction cutoff. The exact source snapshot cutoff is unknown, so the rule avoids assuming all unresolved cases were observed to a common date.

### Cohort scale

The labelable Maharashtra totals from the full yearly cohorts were:

| Filing year | Labelable cases | Over-three-year cases |
|---:|---:|---:|
| 2010 | 844,505 | 299,600 |
| 2011 | 863,394 | 293,271 |
| 2012 | 961,929 | 329,497 |
| 2013 | 1,036,158 | 344,545 |
| 2014 | 1,112,735 | 370,743 |

For 2014, 1,114,737 source rows were in Maharashtra; no duplicate IDs were removed. Rows not labelable under the rule are censored or invalid and are excluded. Detailed counts are in `result/tables/sample_flow_state01_year2010.csv` through `...year2014.csv` and the combined temporal sample-flow table.

## 5. Feature engineering and model inputs

### Model inputs

| Model column | Meaning / encoding |
|---|---|
| `case_type` | Readable case-type category, one-hot encoded. |
| `court_name` | Readable court category, one-hot encoded. |
| `year` | Filing year category, one-hot encoded. |
| `act_section_tokens` | Set of act/section tokens, binary count-vectorized features. |

The categorical encoder groups rare categories (minimum frequency 25 and a maximum of 2,000 categories per field) and handles unseen categories. The act/section vectorizer uses binary token presence, a minimum document frequency of 10, and up to 5,000 tokens. Encoders are fit on training data only, then applied to validation/test data to prevent information leakage from held-out years.

The outcome date, decision date, case identifier, disposition, party/advocate gender indicators, and judge fields are not model predictors. Filing-year `year` is a model input because the task includes filing year, but it also makes the temporal design and year-to-year drift important to discuss.

## 6. Training design and models

### Temporal split

- **Train:** filing years 2010–2012; random 40% sample of eligible cases, 1,067,931 cases in this run.
- **Validation:** 2013; random 20% sample. Used to choose classification thresholds by F1.
- **Test:** 2014; random 20% sample, 222,546 cases. Held out from fitting and threshold selection.
- **k-NN:** due to cost in the sparse, high-dimensional feature space, it used 20,000 training, 5,000 validation, and 5,000 test cases selected stratified by class. Its test metrics are therefore not directly comparable with models scored on all 222,546 test cases.

The sampling fractions reduce memory and runtime. The validation threshold is frozen before scoring the test cohort. Metrics are calculated on the 2014 sample, not on all cases nationally.

### Models selected

1. **Prevalence baseline (`DummyClassifier`)** — always uses the training prevalence as a probability reference. It is a basic chance/reference score.
2. **Historical court/case-type rate baseline** — estimates smoothed delay rates from training cases for each court/case-type combination, backing off to court and overall training rates when needed. This tests how much predictive signal comes from historical group rates.
3. **Class-balanced logistic regression** — linear classifier with balanced class weights; gives a strong, interpretable benchmark for sparse one-hot/token features.
4. **Class-balanced decision tree** — nonlinear tree with depth and minimum-leaf limits to control complexity; class weighting addresses imbalance.
5. **Weighted multinomial Naive Bayes** — probabilistic model for nonnegative sparse feature counts; sample weights balance the two outcome classes.
6. **Sampled k-nearest neighbors (k-NN)** — cosine-distance, 15 neighbors, distance weights; evaluated on the smaller stratified sample because dense similarity calculations on the full sparse feature set are expensive.

Class weighting addresses the approximately one-third positive class prevalence. It does not change the target or make the cohorts balanced for reporting. Thresholds for learned models are chosen on 2013 by maximizing F1, so reported F1/recall/precision reflect that validation-selected operating point. ROC-AUC and PR-AUC use continuous model scores.

## 7. Results and how to read them

The 2014 test sample contains 222,546 cases, of which 73,820 (33.17%) are delayed under the project label rule.

| Model | Test n | ROC-AUC | PR-AUC | F1 | Recall | Precision | Accuracy | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Balanced logistic regression | 222,546 | **0.773** | 0.564 | **0.623** | **0.827** | 0.499 | 0.668 | 0.213 |
| Weighted multinomial Naive Bayes | 222,546 | 0.757 | 0.548 | 0.611 | 0.820 | 0.487 | 0.654 | 0.227 |
| Sampled k-NN | 5,000 | 0.750 | 0.548 | 0.607 | 0.809 | 0.486 | 0.653 | 0.191 |
| Historical court/case-type rate | 222,546 | 0.749 | **0.571** | 0.597 | 0.783 | 0.482 | 0.649 | **0.186** |
| Balanced decision tree | 222,546 | 0.717 | 0.488 | 0.603 | 0.800 | 0.484 | 0.651 | 0.209 |
| Prevalence baseline | 222,546 | 0.500 | 0.332 | 0.000 | 0.000 | 0.000 | 0.668 | 0.222 |

**Which performed better depends on the metric:** logistic regression had the highest ROC-AUC, F1, and recall among the full-test models. The historical rate baseline had the highest PR-AUC and lowest Brier score. The Naive Bayes and decision-tree results are lower on ROC-AUC. k-NN has promising-looking values but a much smaller test set, so treat that comparison cautiously. The plain prevalence baseline's accuracy is just the majority-class rate; its zero F1/recall shows why accuracy alone is not enough.

ROC-AUC measures ranking across thresholds; PR-AUC summarizes precision/recall over thresholds and is useful with an imbalanced target. Recall measures how many labeled delayed cases were identified; precision measures how many predicted-delayed cases were actually labeled delayed. F1 combines precision and recall at the selected threshold. Brier score measures squared probability error (lower is better), but the score alone does not demonstrate good calibration. Review the calibration plot before interpreting outputs as individual case probabilities.

A 100-replicate case-level bootstrap interval for logistic ROC-AUC is 0.771–0.775. It does not account for dependence among cases from the same court. Label sensitivity for 2014 changes the eligible count and delay rate from 1,101,243 / 32.6% under a 2019 listing-date ceiling to 1,112,735 / 33.3% under the 2021 ceiling; this should be discussed as outcome-definition sensitivity.

### Descriptive delay patterns

The test-year pattern tables and plots show observed rates by case type, court, and act/section pair. They are raw subgroup comparisons, not adjusted effects. Small groups can have unstable rates even when a confidence interval is shown; inspect the `cases` column and interval columns before describing high-rate categories. Act/section groups overlap. No patterns by judge religion or judge identity were measured.

## 8. Code files: what runs at each stage

All scripts live under `codebase/`. They can be run individually, but the intended entry point is `bash codebase/run_pipeline.sh` from the project root.

| File | Responsibility |
|---|---|
| `config.py` | Defines project paths, state/year defaults, random seed, output folders, and the last-list plausibility ceiling. |
| `audit_date_coverage.py` | Audits date coverage in the chosen filing-year file; helps expose missing/sentinel/implausible dates. It does not infer the extraction cutoff. |
| `build_cohort.py` | Streams one annual cases file, filters the state, deduplicates case IDs, applies date/target/censoring rules, and writes cohort plus sample-flow table. |
| `build_temporal_cohorts.py` | Repeats cohort building for requested years, combines them into one temporal cohort, and reports annual prevalence. |
| `features_utils.py` | Normalizes act/section codes into stable categorical tokens. |
| `build_temporal_features.py` | Streams the large act/section table, aggregates code tokens per case, joins year-specific case-type and court labels, and writes feature coverage. |
| `label_sensitivity.py` | Recalculates eligible case counts and delay prevalence under alternative last-list date ceilings. |
| `train_temporal_models.py` | Samples chronological train/validation/test cohorts, fits encoders on training only, trains baselines/classifiers, chooses validation thresholds, evaluates test metrics, saves models/predictions/plots/run information. |
| `analyze_patterns.py` | Produces test-year descriptive delay-rate tables and charts by case type, court, and act/section. |
| `bootstrap_temporal_metrics.py` | Estimates case-level bootstrap intervals for model metrics; not court-clustered intervals. |
| `write_conclusion.py` | Reads current metrics/sensitivity/interval tables and writes the generated temporal summary under `conclusion/`. |
| `run_pipeline.py` | Runs audit, cohorts, features, sensitivity, training, pattern analysis, bootstrap, and summary in sequence. |
| `run_pipeline.sh` | Shell wrapper for the full Python pipeline. |
| `requirements.txt` | Python packages: pandas, numpy, scikit-learn, matplotlib, and joblib. |
| `README.md` | Short codebase-specific setup and module overview. |

### Running or rerunning

From the repository root:

```bash
python3 -m pip install -r codebase/requirements.txt
bash codebase/run_pipeline.sh
```

The default is state `01`, train end 2012, validation 2013, and test 2014; years begin at 2010. The pipeline rereads large raw CSVs and regenerates artifacts, so allow sufficient disk space and runtime. Optional arguments can be passed to the Python orchestrator, for example:

```bash
python3 codebase/run_pipeline.py --year-start 2010 --train-end 2012 --validation-year 2013 --test-year 2014 --state-code 01
```

## 9. Results directory: what to open first

### `result/data/`

- `cohort_state01_yearYYYY.csv`: annual state-filtered cohort including raw selected fields and derived outcome/status/date columns.
- `temporal_cohort_state01_2010_2014.csv`: combined annual cohorts before act/section and readable type/court feature joins.
- `temporal_features_state01_2010_2014.csv`: combined cohort plus `act_section_tokens`, `case_type`, and `court_name`. This is the main case-level feature table. It is large and includes identifiers; it is for analysis/reproduction, not for public release without a privacy review.
- `temporal_test_predictions_state01_2010_2014.csv`: true labels and score/prediction columns for the full 2014 test sample.
- `temporal_knn_test_sample_state01_2010_2014.csv`: labels/scores/predictions for the smaller k-NN test subset.
- `temporal_feature_coverage_state01_2010_2014.csv`: counts of cases with act/section records and lookup coverage.

### `result/tables/`

Recommended reading order:

1. `temporal_sample_flow_state01_2010_2014.csv` — source-to-eligible counts by year.
2. `temporal_prevalence_by_year_state01_2010_2014.csv` — sample sizes and delay rates in each split.
3. `label_sensitivity_state01_2010_2014.csv` — effect of alternative listing-date ceilings on labels.
4. `temporal_model_metrics_state01_2010_2012_val2013_test2014.csv` — main comparison table; includes threshold source and test scope.
5. `temporal_metric_bootstrap_ci_state01_2010_2014.csv` — interval estimates for metric uncertainty.
6. `temporal_delay_by_case_type_state01_test2014.csv`, `temporal_delay_by_court_state01_test2014.csv`, and `temporal_delay_by_act_section_state01_test2014.csv` — group counts, delay rates, and 95% intervals.
7. `date_coverage_state01_year2018.csv` — audit diagnostics for later filing-year date fields.
8. `temporal_run_info_state01_2010_2012_val2013_test2014.json` — machine-readable run settings and evaluation description.
9. `sample_flow_state01_year2010.csv` … `sample_flow_state01_year2014.csv` — detailed annual audit counts.

### `result/figures/`

- `temporal_roc_...png`: ROC curves; higher curve/AUC indicates better ranking.
- `temporal_precision_recall_...png`: precision-recall comparison; compare with prevalence because the positive class is about one-third.
- `temporal_calibration_...png`: predicted-score calibration; check whether predicted risk tracks observed frequency.
- `temporal_confusion_matrix_...png`: classification counts for the selected leading full-test model and its validation-selected threshold.
- `temporal_delay_by_case_type_...png`, `temporal_delay_by_court_...png`, `temporal_delay_by_act_section_...png`: descriptive 2014 subgroup delay rates.

### `result/models/`

The four `temporal_*.joblib` artifacts store fitted feature preprocessing and estimators: balanced logistic regression, balanced decision tree, weighted Multinomial Naive Bayes, and sampled k-NN. They are useful for reproduction/scoring with the matching feature columns and package versions; they are not evidence by themselves. There is no saved model for the historical rate and prevalence baselines in the current model directory.

## 10. Suggested path through the repository

1. Read `README.md` for the project summary and `structure.md` for the research questions.
2. Read this file's sections 3–5 and `docs/data_audit.md` to understand source columns, label construction, and censoring.
3. Read `codebase/README.md`, then inspect `run_pipeline.py` to see the stage order.
4. Inspect `build_cohort.py`, `build_temporal_features.py`, and `train_temporal_models.py` for the main data and modeling choices.
5. Start results review with the sample flow and annual prevalence tables; confirm denominators and label prevalence.
6. Review label sensitivity before treating the target as definitive.
7. Compare the main metrics table and calibration plot; use the chosen model only with its threshold and limitations in view.
8. Review group pattern tables alongside sample counts and intervals; phrase findings as descriptive differences.
9. Read the generated conclusion summary, then review `judicial_delay_temporal_paper_draft.pdf` against the supporting tables and figures before submission.

## 11. Limitations to carry into the research paper

- Only Maharashtra is analyzed; the findings are not national estimates.
- Evaluation tests one later filing year and uses random samples from each year for computational scale; external-state and further-year validation remain open.
- The source extraction cutoff is not established. Censoring and the last-list plausibility rule affect which cases can be labeled.
- Raw model scores are not automatically calibrated individual probabilities.
- k-NN uses a smaller test sample and should not be ranked head-to-head with full-sample models without qualification.
- Bootstrap intervals resample cases and do not account for court-level clustering.
- Pattern summaries are unadjusted, potentially sparse subgroup rates; they do not show independent effects or causality.
- Judge religion/identity relationships are not analyzed because the necessary validated attributes/linkage are not part of this project's active evidence.
- DDL data terms and attribution should be checked on the portal before sharing data-derived materials.
