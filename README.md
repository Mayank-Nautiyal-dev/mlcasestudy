# Predicting Three-Year Delay in Maharashtra District Courts

A reproducible machine-learning study of whether case information available at filing can identify Maharashtra district-court cases that take more than three years to reach recorded disposal or remain unresolved beyond that point. The project is aligned with **UN Sustainable Development Goal 16: Peace, Justice and Strong Institutions**.

This repository contains the modeling pipeline, aggregate outputs, charts, data audit, and a journal-style paper draft. The paper is a draft; author/course details and the declarations still need review before submission.

## Research question

Can case type, court, filing year, and the acts/sections listed for a case help predict a three-year delay outcome? A related descriptive analysis compares observed delay rates across case types, courts, and act/section pairs. These comparisons are associations and do not establish causes.

## Data and study subset

The data are from Development Data Lab's Indian Judicial Data, sourced from e-Courts. The DDL portal describes the release as about 81 million district-court cases filed from 2010 through 2018. This project uses **Maharashtra (state code `01`)**, filing years **2010–2014**. The raw DDL data are not included in this checkout; `datasetddl/` currently contains only a source note. Download the CSV release from the [DDL Judicial Data portal](https://www.devdatalab.org/judicial-data), follow its current terms and attribution instructions, and place it in this layout:

```text
datasetddl/csv/
├── cases/cases/cases_2010.csv ... cases_2018.csv
├── acts_sections.csv
└── keys/keys/
    ├── act_key.csv
    ├── section_key.csv
    ├── type_name_key.csv
    ├── disp_name_key.csv
    ├── purpose_name_key.csv
    ├── cases_court_key.csv
    ├── cases_district_key.csv
    └── cases_state_key.csv
```

The DDL portal currently lists the data under the **CC BY-NC-SA 4.0** license for non-commercial use. Check the portal for current requirements before sharing any data-derived material. The raw case files and derived case-level CSVs are excluded by `.gitignore`; the aggregate results and figures are retained in the repository.

### Fields in the project subset and fields used

The cohort builder reads the following columns from each annual case CSV. The DDL release may contain additional fields; this pipeline does not ingest every source column.

| Annual case field(s) | Role here |
|---|---|
| `ddl_case_id`, `cino` | Source identifiers for joining and traceability; not predictors. |
| `year`, `state_code`, `dist_code`, `court_no` | Filing-year split and state/court selection; codes are joined to readable court labels. |
| `type_name` | Coded case type, mapped to `case_type`, a model feature. |
| `purpose_name` | Retained in the cohort for context; not a model feature. |
| `disp_name` | Joined to the year-specific disposition key to determine whether a missing decision is a missing disposition; not a model feature. |
| `date_of_filing`, `date_of_decision`, `date_last_list` | Used to build the three-year outcome and determine censoring; never model features. |
| `judge_position` | Retained in the cohort but not used for judge analysis or as a model feature. It is a position/designation field, not a judge name or religion. |
| `female_defendant`, `female_petitioner`, `female_adv_def`, `female_adv_pet` | Present in the selected case columns but excluded from modeling and analysis. |

The separate `acts_sections.csv` contributes `ddl_case_id`, `act`, and `section`. Distinct act/section pairs are aggregated per case as binary tokens in `act_section_tokens`. Lookup files supply readable case-type, court, act, and section labels; the disposition lookup also supports the pending-case label rule.

**Actual classifier inputs:** `case_type`, `court_name`, filing `year`, and `act_section_tokens`. Case IDs, outcome dates, disposition, hearing purpose, judge fields, and party/advocate gender are not model predictors. Judge religion, judge identity, judge-party relationships, and gender effects are outside the active scope; the proposed judge relationship task was cancelled.

### Outcome definition

A case with a valid decision date is labeled delayed when the decision occurs later than the third calendar anniversary of filing. If the decision is missing, the case is labeled delayed only when its disposition lookup marks disposition as missing and a plausible last-list date is after that anniversary. The `2021-12-31` last-list ceiling is a data-quality rule, **not** a known source extraction cutoff. Cases without enough follow-up are censored and excluded.

This operational outcome includes both recorded late disposals and cases documented as still pending after three years. It should not be interpreted as a prediction of eventual disposal duration for every case.

## Evaluation design

- **Training:** filing years 2010–2012; random 40% sample of labelable cases (`n = 1,067,931`).
- **Validation:** 2013; random 20% sample (`n = 207,233`) to select classification thresholds by F1.
- **Test:** 2014; random 20% sample (`n = 222,546`).
- Encoders are fitted on training data only. The 2014 test year does not determine model fitting or thresholds.
- k-nearest neighbors (k-NN) uses stratified samples of 20,000 training, 5,000 validation, and 5,000 test cases because of the sparse, high-dimensional features. Its scores are not directly comparable with full-test models.

## Test results

The test sample has a **33.17%** positive rate. Threshold-dependent scores use thresholds chosen on the 2013 validation cohort. ROC-AUC and PR-AUC use model scores.

| Model | Test n | ROC-AUC | PR-AUC | F1 | Recall | Precision | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced logistic regression | 222,546 | **0.773** | 0.564 | **0.623** | **0.827** | 0.499 | 0.213 |
| Weighted multinomial Naive Bayes | 222,546 | 0.757 | 0.548 | 0.611 | 0.820 | 0.487 | 0.227 |
| Sampled k-NN | 5,000 | 0.750 | 0.548 | 0.607 | 0.809 | 0.486 | 0.191 |
| Historical court/case-type rate baseline | 222,546 | 0.749 | **0.571** | 0.597 | 0.783 | 0.482 | **0.186** |
| Balanced decision tree | 222,546 | 0.717 | 0.488 | 0.603 | 0.800 | 0.484 | 0.209 |
| Prevalence baseline | 222,546 | 0.500 | 0.332 | 0.000 | 0.000 | 0.000 | 0.222 |

Logistic regression led on ROC-AUC, F1, and recall among the full-test models. The historical court/case-type baseline led on PR-AUC and Brier score, so the classifier does not beat every simple benchmark on every metric. The logistic ROC-AUC case-level bootstrap 95% interval is 0.771–0.775 (100 resamples); this does not account for court-level clustering. The 2014 label sensitivity analysis gives a delay rate from 32.6% to 33.3% across tested last-list ceilings.

The pattern tables and figures summarize delay rates by case type, court, and act/section in the full labelable 2014 cohort. They are unadjusted descriptive comparisons, not causal estimates. Act/section groups overlap because one case may have several pairs.

## Models

The code trains a prevalence baseline, a historical court/case-type rate baseline, class-balanced logistic regression, a class-balanced decision tree, weighted Multinomial Naive Bayes, and sampled cosine-distance k-NN. Categorical predictors use one-hot encoding with infrequent-category handling; act/section pairs use binary token features. Class weighting/sample weights address class imbalance. Validation-year F1 selects learned-model thresholds.

## Repository guide

| Path | Contents |
|---|---|
| `codebase/` | Cohort/date audits, feature building, model training, pattern analysis, bootstrap, summary writer, dependencies, and pipeline entry point. See `codebase/README.md`. |
| `datasetddl/README.md` | DDL source link and note that raw data must be obtained separately. |
| `docs/data_audit.md` | Schema, cohort counts, outcome construction, source-date audit, and limitations. |
| `result/tables/` | Aggregate cohort flow, model metrics, sensitivity, bootstrap intervals, and subgroup delay rates. |
| `result/figures/` | ROC, precision-recall, calibration, confusion matrix, and descriptive delay charts. |
| `result/data/` | Regenerated case-level cohorts, features, and prediction files; ignored by Git because they are large and contain identifiers. |
| `result/models/` | Regenerated fitted model pipelines; ignored by Git as binary outputs. |
| `conclusion/` | Generated temporal evaluation summary. |
| `structure.md` | Research questions, scope, methods, and limitations. |
| `detailedsteps.md` | Full file-by-file walkthrough of the data, code, models, results, and suggested reading order. |
| `judicial_delay_temporal_paper_draft.pdf` | Journal-style paper draft based on the current results. It includes student name, roll number, and affiliation; course, supervisor, and email fields remain placeholders. |

## Reproduce the analysis

Python 3.10 or newer is recommended. From the repository root, install dependencies and run:

```bash
python3 -m pip install -r codebase/requirements.txt
bash codebase/run_pipeline.sh
```

The default pipeline uses state `01`, training years 2010–2012, validation year 2013, and test year 2014. It reads raw DDL files from `datasetddl/csv/` and regenerates cohorts, features, models, tables, charts, and the conclusion. The raw files are large, and the feature-building stage requires substantial disk space. To change years or state, use `codebase/run_pipeline.py --help`.

## Git and publishing note

`.gitignore` excludes raw DDL downloads, generated case-level CSVs, serialized models, local environments, and editor files while keeping aggregate tables, figures, documentation, and the paper draft. The original `.git` history contains multi-gigabyte case-level CSV blobs, so do not push that history. A clean replacement Git directory named `.git-publish` is prepared for a data-free initial commit; the old `.git` is left untouched as a local backup because this workspace restricts writes to it. From the project root, use:

```bash
git --git-dir=.git-publish --work-tree=. push -u origin main
```

The new history includes code, documentation, aggregate tables/figures, and the paper draft; it excludes raw DDL downloads and regenerated case-level data/model binaries. The DDL source files are absent from this checkout and should remain outside Git.

## Limitations

- This is one-state evidence, not a national estimate.
- The exact DDL extraction cutoff is undocumented; the pending-case label and censoring rules affect the sample.
- Evaluation tests one later filing year. Generalization to other states and later years is not established.
- Model scores are not validated as calibrated individual-case probabilities.
- k-NN uses a smaller test set. Bootstrap intervals resample cases rather than courts.
- The subgroup rates are not adjusted for case mix and do not identify independent or causal effects.
- Judge religion and judge-case relationships are outside the active scope.

For the detailed explanation of every script and output, start with [`detailedsteps.md`](detailedsteps.md). For a quick summary of the analysis findings, read [`conclusion/temporal_evaluation_state01_2010_2012_val2013_test2014.md`](conclusion/temporal_evaluation_state01_2010_2012_val2013_test2014.md).
