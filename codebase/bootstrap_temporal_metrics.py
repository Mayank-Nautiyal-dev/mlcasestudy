"""Case-level bootstrap confidence intervals for temporal test metrics."""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score

from config import DATA_OUT, STATE_CODE, TABLES_OUT


PROBABILITY_COLUMNS = {
    "prevalence_baseline": "prevalence_baseline_probability",
    "historical_court_case_type_rate": "historical_rate_probability",
    "logistic_regression_balanced": "logistic_regression_balanced_probability",
    "decision_tree_balanced": "decision_tree_balanced_probability",
    "multinomial_naive_bayes": "multinomial_naive_bayes_probability",
}


def run(year_start=2010, test_year=2014, state_code=STATE_CODE, replicates=200):
    state_code = str(state_code).zfill(2)
    path = DATA_OUT / f"temporal_test_predictions_state{state_code}_{year_start}_{test_year}.csv"
    predictions = pd.read_csv(path)
    y = predictions["y_true"].to_numpy(dtype=int)
    rng = np.random.default_rng(2026)
    result = []
    metric_fns = {
        "roc_auc": lambda labels, scores, threshold: roc_auc_score(labels, scores),
        "pr_auc": lambda labels, scores, threshold: average_precision_score(labels, scores),
        "f1": lambda labels, scores, threshold: f1_score(labels, scores >= threshold, zero_division=0),
        "recall": lambda labels, scores, threshold: recall_score(labels, scores >= threshold, zero_division=0),
    }
    metrics_table = pd.read_csv(
        TABLES_OUT / f"temporal_model_metrics_state{state_code}_{year_start}_2012_val2013_test{test_year}.csv"
    )
    for model, probability_col in PROBABILITY_COLUMNS.items():
        if probability_col not in predictions:
            continue
        scores = predictions[probability_col].to_numpy(dtype=float)
        metric_row = metrics_table.loc[metrics_table["model"].eq(model)].iloc[0]
        threshold = float(metric_row.get("threshold_from_validation", 0.5))
        for metric, fn in metric_fns.items():
            samples = np.empty(replicates, dtype=float)
            for i in range(replicates):
                idx = rng.integers(0, len(y), size=len(y))
                samples[i] = fn(y[idx], scores[idx], threshold)
            result.append({
                "model": model,
                "metric": metric,
                "point_estimate": float(fn(y, scores, threshold)),
                "ci_2_5": float(np.quantile(samples, 0.025)),
                "ci_97_5": float(np.quantile(samples, 0.975)),
                "replicates": replicates,
                "test_n": len(y),
            })
    output = pd.DataFrame(result)
    target = TABLES_OUT / f"temporal_metric_bootstrap_ci_state{state_code}_{year_start}_{test_year}.csv"
    output.to_csv(target, index=False)
    print(output.to_string(index=False))
    print("Intervals resample cases independently; they do not account for court-level clustering.")
    print(f"Wrote {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-start", type=int, default=2010)
    parser.add_argument("--test-year", type=int, default=2014)
    parser.add_argument("--state-code", default=STATE_CODE)
    parser.add_argument("--replicates", type=int, default=200)
    args = parser.parse_args()
    run(args.year_start, args.test_year, args.state_code, args.replicates)
