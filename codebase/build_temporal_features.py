"""Join large act/section data and codebook labels to a multi-year cohort."""
import argparse

import pandas as pd

from config import ACTS_FILE, DATA_OUT, KEYS_DIR, STATE_CODE
from features_utils import code_token


def normalize_code(series, width=0):
    result = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    if width:
        result = result.str.zfill(width)
    return result


def build(year_start=2010, year_end=2014, state_code=STATE_CODE):
    state_code = str(state_code).zfill(2)
    cohort_path = DATA_OUT / f"temporal_cohort_state{state_code}_{year_start}_{year_end}.csv"
    if not cohort_path.exists():
        raise FileNotFoundError(f"Run build_temporal_cohorts.py first: {cohort_path}")
    cohort = pd.read_csv(cohort_path, dtype={"ddl_case_id": str}, low_memory=False)
    cohort["year"] = normalize_code(cohort["year"])
    cohort["type_name"] = normalize_code(cohort["type_name"])
    cohort["state_code"] = normalize_code(cohort["state_code"], 2)
    cohort["dist_code"] = normalize_code(cohort["dist_code"], 2)
    cohort["court_no"] = normalize_code(cohort["court_no"], 2)
    ids = set(cohort["ddl_case_id"].dropna())
    token_map = {}
    matched_rows = 0
    scanned_rows = 0
    chunk_number = 0

    for chunk in pd.read_csv(
        ACTS_FILE, usecols=["ddl_case_id", "act", "section"], dtype=str,
        chunksize=1_000_000, low_memory=False,
    ):
        chunk_number += 1
        scanned_rows += len(chunk)
        selected = chunk.loc[chunk["ddl_case_id"].isin(ids)]
        matched_rows += len(selected)
        for case_id, act, section in selected.itertuples(index=False, name=None):
            pair = "_".join(filter(None, (code_token(act, "A"), code_token(section, "S"))))
            if pair:
                token_map.setdefault(case_id, set()).add(pair)
        if chunk_number % 3 == 0:
            print(f"Scanned {scanned_rows:,} act/section rows; matched {matched_rows:,} selected rows", flush=True)

    cohort["act_section_tokens"] = cohort["ddl_case_id"].map(
        lambda case_id: " ".join(sorted(token_map.get(case_id, set())))
    )

    type_lookup = pd.read_csv(KEYS_DIR / "type_name_key.csv", dtype=str, low_memory=False)
    type_lookup["year"] = normalize_code(type_lookup["year"])
    type_lookup["type_name"] = normalize_code(type_lookup["type_name"])
    type_lookup = type_lookup.drop_duplicates(["year", "type_name"], keep="first")
    cohort = cohort.merge(
        type_lookup[["year", "type_name", "type_name_s"]],
        on=["year", "type_name"], how="left", validate="many_to_one",
    )
    cohort["case_type"] = cohort["type_name_s"].fillna(cohort["type_name"].astype(str))

    court_lookup = pd.read_csv(KEYS_DIR / "cases_court_key.csv", dtype=str, low_memory=False)
    for col, width in [("year", 0), ("state_code", 2), ("dist_code", 2), ("court_no", 2)]:
        court_lookup[col] = normalize_code(court_lookup[col], width)
    court_lookup = court_lookup.drop_duplicates(
        ["year", "state_code", "dist_code", "court_no"], keep="first"
    )
    cohort = cohort.merge(
        court_lookup[["year", "state_code", "dist_code", "court_no", "court_name"]],
        on=["year", "state_code", "dist_code", "court_no"],
        how="left", validate="many_to_one",
    )
    cohort["court_name"] = cohort["court_name"].fillna(cohort["court_id"])

    output = DATA_OUT / f"temporal_features_state{state_code}_{year_start}_{year_end}.csv"
    cohort.to_csv(output, index=False)
    coverage = pd.DataFrame([
        ("unique_cases", len(cohort)),
        ("cases_with_act_section_record", int(cohort["act_section_tokens"].ne("").sum())),
        ("cases_without_act_section_record", int(cohort["act_section_tokens"].eq("").sum())),
        ("matched_act_section_rows", matched_rows),
        ("missing_case_type_label", int(cohort["type_name_s"].isna().sum())),
        ("missing_court_name", int(cohort["court_name"].isna().sum())),
    ], columns=["feature_coverage_measure", "count"])
    coverage.to_csv(
        output.with_name(f"temporal_feature_coverage_state{state_code}_{year_start}_{year_end}.csv"),
        index=False,
    )
    print(f"Temporal features written: {output}")
    print(coverage.to_string(index=False))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-start", type=int, default=2010)
    parser.add_argument("--year-end", type=int, default=2014)
    parser.add_argument("--state-code", default=STATE_CODE)
    args = parser.parse_args()
    build(args.year_start, args.year_end, args.state_code)
