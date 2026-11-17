# Temporal evaluation outputs

Current results are from the Maharashtra (state code `01`) temporal experiment: train on filing years 2010–2012, tune thresholds on 2013, and evaluate on a sampled 2014 test cohort. The generated narrative summary is in `../conclusion/temporal_evaluation_state01_2010_2012_val2013_test2014.md`.

## Output folders

- `data/`: annual and combined labeled cohorts, derived case-level temporal features, test predictions, and the smaller k-NN test sample.
- `models/`: fitted preprocessing and estimator pipelines from the temporal run.
- `tables/`: yearly sample flow and prevalence; date audit; listing-date label sensitivity; temporal model metrics and bootstrap intervals; test-year delay rates by case type, court, and act/section.
- `figures/`: temporal ROC, precision-recall, calibration, confusion matrix, and test-year descriptive delay charts.

The pattern tables are unadjusted descriptions of the labelable 2014 cohort. For act/section, a case with multiple distinct pairs contributes once to each pair, so groups overlap. These comparisons do not establish causal effects. The judge-identity task was cancelled; judge-related files from the earlier pilot are not current findings.

Sampling and label definitions, including censoring limitations, are documented in the root `README.md` and `docs/data_audit.md`.
