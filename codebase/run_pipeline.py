"""Run the chronological cohort, models, validation, patterns, and summary."""
import argparse

from analyze_patterns import analyze
from audit_date_coverage import audit as audit_dates
from bootstrap_temporal_metrics import run as bootstrap_metrics
from build_temporal_cohorts import run as build_cohorts
from build_temporal_features import build as build_features
from label_sensitivity import run as analyze_label_sensitivity
from train_temporal_models import run as train_models
from write_conclusion import write as write_summary


def run(year_start, train_end, validation_year, test_year, state_code):
    years = tuple(range(year_start, test_year + 1))
    audit_dates(2018, state_code)
    build_cohorts(years, state_code)
    build_features(year_start, test_year, state_code)
    analyze_label_sensitivity(year_start, test_year, state_code)
    train_models(year_start, train_end, validation_year, test_year, state_code)
    analyze(year_start, test_year, state_code)
    bootstrap_metrics(year_start, test_year, state_code)
    write_summary(year_start, train_end, validation_year, test_year, state_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-start", type=int, default=2010)
    parser.add_argument("--train-end", type=int, default=2012)
    parser.add_argument("--validation-year", type=int, default=2013)
    parser.add_argument("--test-year", type=int, default=2014)
    parser.add_argument("--state-code", default="01")
    args = parser.parse_args()
    run(args.year_start, args.train_end, args.validation_year, args.test_year, args.state_code)
