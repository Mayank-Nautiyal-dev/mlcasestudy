"""Build a labeled, one-row-per-case temporal cohort from a yearly CSV."""
import argparse
import csv
import pandas as pd

from config import CASE_DIR, DATA_OUT, FILING_YEAR, KEYS_DIR, MAX_PLAUSIBLE_LISTING_DATE, STATE_CODE, TABLES_OUT


CASE_COLUMNS = [
    "ddl_case_id", "year", "state_code", "dist_code", "court_no", "cino",
    "judge_position", "type_name", "purpose_name", "disp_name",
    "date_of_filing", "date_of_decision", "date_last_list", "female_defendant",
    "female_petitioner", "female_adv_def", "female_adv_pet",
]


def clean_code(value):
    if pd.isna(value):
        return "UNKNOWN"
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value or "UNKNOWN"


def build(year=FILING_YEAR, state_code=STATE_CODE):
    source = CASE_DIR / f"cases_{year}.csv"
    if not source.exists():
        raise FileNotFoundError(f"Case file not found: {source}")

    state_code = str(state_code).zfill(2)
    chunks = []
    source_rows = 0
    state_rows = 0
    for part in pd.read_csv(
        source, usecols=CASE_COLUMNS, dtype=str, chunksize=200_000,
        low_memory=False,
    ):
        source_rows += len(part)
        part["state_code"] = part["state_code"].map(clean_code).str.zfill(2)
        part = part.loc[part["state_code"].eq(state_code)].copy()
        state_rows += len(part)
        if len(part):
            chunks.append(part)

    if not chunks:
        raise ValueError(f"No cases found for state {state_code}, year {year}")
    cohort = pd.concat(chunks, ignore_index=True)
    duplicate_rows = int(cohort.duplicated("ddl_case_id", keep="first").sum())
    cohort = cohort.drop_duplicates("ddl_case_id", keep="first").copy()

    filing = pd.to_datetime(cohort["date_of_filing"], errors="coerce")
    decision_raw = cohort["date_of_decision"].astype("string").str.strip()
    sentinel = decision_raw.str.startswith("5000-", na=False)
    decision = pd.to_datetime(decision_raw.mask(sentinel), errors="coerce")
    last_raw = cohort["date_last_list"].astype("string").str.strip()
    last_sentinel = last_raw.str.startswith("5000-", na=False)
    last_list = pd.to_datetime(last_raw.mask(last_sentinel), errors="coerce")
    invalid_filing = filing.isna()
    decision_before_filing = decision.notna() & filing.notna() & decision.lt(filing)
    valid_filing = ~invalid_filing
    valid_decision = decision.notna() & ~decision_before_filing
    deadline = filing + pd.DateOffset(years=3)

    # DDL's year-specific disposition lookup includes a code whose label says
    # the disposition variable is missing. For those unresolved records, a
    # valid last-list date after the three-year anniversary is evidence that
    # the case had not reached recorded disposal by that point. Other cases
    # without a decision date remain censored/ambiguous.
    disp_lookup = pd.read_csv(KEYS_DIR / "disp_name_key.csv", dtype=str, low_memory=False)
    disp_lookup = disp_lookup.loc[disp_lookup["year"].eq(str(year))]
    missing_disp_codes = set(
        disp_lookup.loc[
            disp_lookup["disp_name_s"].str.contains("missing", case=False, na=False),
            "disp_name",
        ].dropna().astype(str)
    )
    disp_code = cohort["disp_name"].astype("string").str.replace(r"\.0$", "", regex=True)
    no_disposition = disp_code.isin(missing_disp_codes)
    plausible_last = (
        last_list.notna() & last_list.ge(filing) &
        last_list.le(pd.Timestamp(MAX_PLAUSIBLE_LISTING_DATE))
    )
    pending_over_three = (
        valid_filing & decision.isna() & no_disposition & plausible_last &
        last_list.gt(deadline)
    )
    disposed_valid = valid_filing & valid_decision
    eligible = disposed_valid | pending_over_three
    cohort["delay_over_3_years"] = pd.Series(pd.NA, index=cohort.index, dtype="Int64")
    cohort.loc[disposed_valid, "delay_over_3_years"] = decision.loc[disposed_valid].gt(
        deadline.loc[disposed_valid]
    ).astype(int)
    cohort.loc[pending_over_three, "delay_over_3_years"] = 1
    cohort["outcome_status"] = "censored_or_invalid"
    cohort.loc[disposed_valid & decision.le(deadline), "outcome_status"] = "disposed_within_3_years"
    cohort.loc[disposed_valid & decision.gt(deadline), "outcome_status"] = "disposed_after_3_years"
    cohort.loc[pending_over_three, "outcome_status"] = "pending_observed_after_3_years"
    cohort["filing_date_parsed"] = filing.dt.strftime("%Y-%m-%d")
    cohort["decision_date_parsed"] = decision.dt.strftime("%Y-%m-%d")
    cohort["last_list_date_parsed"] = last_list.dt.strftime("%Y-%m-%d")
    cohort["duration_days"] = (decision - filing).dt.days.astype("Int64")

    for field in ("state_code", "dist_code", "court_no", "type_name", "year"):
        if field in cohort.columns:
            cohort[field] = cohort[field].map(clean_code)
    cohort["court_id"] = (
        cohort["state_code"].fillna("UNKNOWN") + "_" +
        cohort["dist_code"].fillna("UNKNOWN") + "_" +
        cohort["court_no"].fillna("UNKNOWN")
    )

    # Preserve IDs locally for linkage; do not include identifying IDs in public
    # aggregate tables or model features.
    output = DATA_OUT / f"cohort_state{state_code}_year{year}.csv"
    cohort.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)

    flow = pd.DataFrame([
        ("all_source_rows_in_filing_year", source_rows),
        ("rows_in_selected_state", state_rows),
        ("duplicate_case_rows_removed", duplicate_rows),
        ("unique_cases_after_deduplication", len(cohort)),
        ("invalid_or_missing_filing_date", int(invalid_filing.sum())),
        ("missing_or_sentinel_decision_date", int(decision.isna().sum())),
        ("decision_before_filing", int(decision_before_filing.sum())),
        ("pending_cases_observed_after_three_years", int(pending_over_three.sum())),
        ("pending_cases_censored_before_three_years_or_invalid_listing", int((valid_filing & decision.isna() & no_disposition & ~pending_over_three).sum())),
        ("missing_decision_with_nonmissing_disposition", int((valid_filing & decision.isna() & ~no_disposition).sum())),
        ("eligible_disposed_or_confirmed_pending_cases", int(eligible.sum())),
        ("excluded_censored_or_invalid_cases", int((~eligible).sum())),
        ("eligible_delay_over_3_years", int((cohort["delay_over_3_years"] == 1).sum())),
        ("eligible_disposed_within_3_years", int((cohort["delay_over_3_years"] == 0).sum())),
    ], columns=["sample_flow_step", "case_count"])
    flow.to_csv(TABLES_OUT / f"sample_flow_state{state_code}_year{year}.csv", index=False)

    print(f"Cohort written: {output}")
    print(flow.to_string(index=False))
    print("Unresolved cases are labeled delayed only when disposition is coded missing and a plausible last-list date is after the three-year anniversary; other unresolved cases remain censored.")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=FILING_YEAR)
    parser.add_argument("--state-code", default=STATE_CODE)
    args = parser.parse_args()
    build(args.year, args.state_code)
