# Temporal judicial-delay pipeline

The active pipeline builds a Maharashtra 2010–2014 cohort, trains on earlier filing years, selects thresholds on 2013, tests on a 2014 sample, and writes metrics, figures, delay-pattern tables, and a generated conclusion. It does not perform judge or party identity analysis.

## Run

From the project root:

```bash
bash codebase/run_pipeline.sh
```

The default is state `01`, train 2010–2012, validation 2013, test 2014. Arguments can be passed through to `run_pipeline.py`, for example:

```bash
bash codebase/run_pipeline.sh --year-start 2010 --train-end 2012 --validation-year 2013 --test-year 2014 --state-code 01
```

Dependencies are listed in `requirements.txt`. Install with `python3 -m pip install -r codebase/requirements.txt` if needed.

## Modules

| File | Purpose |
|---|---|
| `config.py` | Shared paths, state/year defaults, and the 2021 listing-date plausibility ceiling. |
| `audit_date_coverage.py` | Summarizes date coverage in the latest filing cohort; date maxima are diagnostic and do not establish the source extraction cutoff. |
| `build_cohort.py` | Streams one year's case file, filters state, deduplicates IDs, validates dates, applies the three-year/censoring rule, and writes sample flow. |
| `build_temporal_cohorts.py` | Builds and combines the train, validation, and test filing-year cohorts. |
| `features_utils.py` | Converts act/section codes to stable per-case feature tokens. |
| `build_temporal_features.py` | Streams the large act/section table and joins year-specific case-type and court labels without duplicating cases. |
| `label_sensitivity.py` | Recomputes eligible cases and delay prevalence under alternative listing-date ceilings. |
| `train_temporal_models.py` | Fits train-only encoders and baseline, logistic, decision-tree, Naive Bayes, and sampled k-NN models; tunes thresholds on validation; scores the later test cohort. |
| `analyze_patterns.py` | Produces case-type, act/section, and court delay-rate tables and charts for the test filing year. |
| `bootstrap_temporal_metrics.py` | Generates case-level bootstrap intervals for full-test-sample metrics. Intervals do not account for court clustering. |
| `write_conclusion.py` | Generates the temporal evaluation summary from result tables. |
| `run_pipeline.py` | Runs all stages in order. |
| `run_pipeline.sh` | Shell entry point. |

## Target and censoring

A valid decision after the third calendar anniversary is positive; a valid decision on or before it is negative. If the decision is missing and the disposition lookup marks the disposition variable as missing, a valid last-list date after the three-year anniversary is sufficient to label the case positive. `2021-12-31` is used as a plausibility ceiling for that listing date, not as an assumed source snapshot. Other unresolved cases are censored and excluded.

## Sampling and evaluation

The classifier uses a 40% random sample of eligible training cases from 2010–2012, a 20% sample of 2013 for threshold tuning, and a 20% sample of 2014 for the test metrics. The act/section and category encoders are fitted on training data only. k-NN is further restricted to 20,000 training, 5,000 validation, and 5,000 test cases due to its cost in the sparse feature space; its metrics use a smaller test set.

The generated model probabilities should be checked against the calibration plot and Brier score before being interpreted as individual case risks.
