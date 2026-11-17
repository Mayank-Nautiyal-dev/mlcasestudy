"""Generate the temporal-evaluation summary from produced artifacts."""
import argparse
import json

import pandas as pd

from config import CONCLUSION_DIR, STATE_CODE, TABLES_OUT


def write(year_start=2010, train_end=2012, validation_year=2013, test_year=2014, state_code=STATE_CODE):
    state_code = str(state_code).zfill(2)
    suffix = f"state{state_code}_{year_start}_{train_end}_val{validation_year}_test{test_year}"
    metrics_path = TABLES_OUT / f"temporal_model_metrics_{suffix}.csv"
    info_path = TABLES_OUT / f"temporal_run_info_{suffix}.json"
    sensitivity_path = TABLES_OUT / f"label_sensitivity_state{state_code}_{year_start}_{test_year}.csv"
    bootstrap_path = TABLES_OUT / f"temporal_metric_bootstrap_ci_state{state_code}_{year_start}_{test_year}.csv"
    metrics = pd.read_csv(metrics_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    best_auc = metrics.sort_values("roc_auc", ascending=False).iloc[0]
    best_pr = metrics.sort_values("pr_auc", ascending=False).iloc[0]
    out = [
        f"# Temporal model evaluation: Maharashtra, test filing year {test_year}",
        "",
        "## Evaluation design",
        "",
        f"- Training cohorts: {year_start}–{train_end}; model training used a random 40% sample of eligible cases from those years.",
        f"- Validation cohort: {validation_year}; threshold tuning used a random 20% sample.",
        f"- Held-out test cohort: {test_year}; metrics use a random 20% sample, with n={int(info['test_n']):,}.",
        "- Categories and act/section encoders were fitted on training years only. Thresholds for learned models were selected by validation-year F1 and then frozen for test scoring.",
        "",
        "The results estimate performance on later filing-year cases with observed labels in this state. They do not establish performance across India or future source releases.",
        "",
        "## Outcome definition and censoring",
        "",
        "A case is positive if its recorded decision is later than the third calendar anniversary of filing. A case with no recorded decision is also labeled positive only when the year-specific disposition lookup marks disposition as missing and a plausible `date_last_list` is after that anniversary. The listing-date plausibility ceiling is 2021-12-31, used to reject implausible far-future dates; it is not treated as a source snapshot cutoff. Cases without sufficient recorded follow-up remain censored and are excluded.",
        "",
        "## Test results",
        "",
        f"The best ROC-AUC is **{best_auc['model']}** (ROC-AUC {best_auc['roc_auc']:.3f}, PR-AUC {best_auc['pr_auc']:.3f}, F1 {best_auc['f1']:.3f}, recall {best_auc['recall']:.3f}, precision {best_auc['precision']:.3f}). Its threshold ({best_auc['threshold_from_validation']:.3f}) was selected on validation data.",
        f"The highest PR-AUC is **{best_pr['model']}** at {best_pr['pr_auc']:.3f}. k-NN was trained on 20,000 and evaluated on 5,000 sampled cases because of the high-dimensional sparse feature space; its metrics are not directly comparable with full test-sample metrics.",
        "",
        "| Model | Test n | ROC-AUC | PR-AUC | F1 | Recall | Precision | Brier | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        out.append(
            f"| {row['model']} | {int(row['test_n']):,} | {row['roc_auc']:.3f} | {row['pr_auc']:.3f} | {row['f1']:.3f} | {row['recall']:.3f} | {row['precision']:.3f} | {row['brier_score']:.3f} | {row['threshold_from_validation']:.3f} |"
        )
    out.extend([
        "",
        "Probability calibration is reported with Brier scores and a reliability plot. Raw model scores should not be described as literal case-level probabilities unless calibration is adequate.",
        "",
        "## Label sensitivity",
    ])
    if sensitivity_path.exists():
        sens = pd.read_csv(sensitivity_path)
        test = sens.loc[sens["filing_year"].eq(test_year)].sort_values("listing_date_ceiling")
        for _, row in test.iterrows():
            out.append(f"- Listing-date ceiling {row['listing_date_ceiling']}: {int(row['eligible_cases']):,} eligible; delay rate {row['delay_rate']:.1%}.")
    else:
        out.append("The listing-date ceiling sensitivity table was not generated.")
    out.extend([
        "",
        "## Uncertainty and limits",
        "",
    ])
    if bootstrap_path.exists():
        ci = pd.read_csv(bootstrap_path)
        best_auc_ci = ci.loc[(ci["model"] == best_auc["model"]) & (ci["metric"] == "roc_auc")]
        if not best_auc_ci.empty:
            row = best_auc_ci.iloc[0]
            out.append(f"A case-level bootstrap 95% interval for the best model's ROC-AUC is {row['ci_2_5']:.3f}–{row['ci_97_5']:.3f} ({int(row['replicates'])} replicates). This interval does not account for clustering by court.")
    out.extend([
        "The listing-date rule and 2021 plausibility ceiling affect which pending cases are labelable. The DDL portal does not provide an exact extraction cutoff, and date fields contain malformed far-future values. Results therefore need a sensitivity review and should not be framed as a definitive all-case risk estimate.",
        "",
        "The judge-relationship analysis has been removed from the active project scope at the user's direction. No judge-related results are included here.",
        "",
        "## Reproducibility files",
        "",
        "Code is under `codebase/`. Cohort, feature, prediction, model, metric, sensitivity, and chart artifacts are under `result/`. Case-type, court, and act/section descriptive tables summarize the held-out test filing year.",
    ])
    output = CONCLUSION_DIR / f"temporal_evaluation_state{state_code}_{year_start}_{train_end}_val{validation_year}_test{test_year}.md"
    output.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Temporal summary written: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-start", type=int, default=2010)
    parser.add_argument("--train-end", type=int, default=2012)
    parser.add_argument("--validation-year", type=int, default=2013)
    parser.add_argument("--test-year", type=int, default=2014)
    parser.add_argument("--state-code", default=STATE_CODE)
    args = parser.parse_args()
    write(args.year_start, args.train_end, args.validation_year, args.test_year, args.state_code)
