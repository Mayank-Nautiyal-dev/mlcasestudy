"""Recompute unresolved-case labels under alternative listing-date plausibility ceilings."""
import argparse

import pandas as pd

from config import DATA_OUT, KEYS_DIR, STATE_CODE, TABLES_OUT


CAPS = ["2019-12-31", "2020-12-31", "2021-12-31", "2025-12-31"]


def run(year_start=2010, year_end=2014, state_code=STATE_CODE):
    state_code = str(state_code).zfill(2)
    path = DATA_OUT / f"temporal_cohort_state{state_code}_{year_start}_{year_end}.csv"
    usecols = [
        "year", "date_of_filing", "date_of_decision", "last_list_date_parsed",
        "disp_name", "delay_over_3_years",
    ]
    frame = pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False)
    frame["year"] = frame["year"].astype(str)
    filing = pd.to_datetime(frame["date_of_filing"], errors="coerce")
    raw_decision = frame["date_of_decision"].astype("string").str.strip()
    decision = pd.to_datetime(
        raw_decision.mask(raw_decision.str.startswith("5000-", na=False)), errors="coerce"
    )
    last_list = pd.to_datetime(frame["last_list_date_parsed"], errors="coerce")
    deadline = filing + pd.DateOffset(years=3)
    disp_lookup = pd.read_csv(KEYS_DIR / "disp_name_key.csv", dtype=str, low_memory=False)
    code_to_missing = {}
    for year in sorted(frame["year"].dropna().unique()):
        part = disp_lookup.loc[disp_lookup["year"].eq(year)]
        code_to_missing[year] = set(
            part.loc[part["disp_name_s"].str.contains("missing", case=False, na=False), "disp_name"].dropna()
        )
    results = []
    for cap in CAPS:
        valid_filing = filing.notna()
        valid_disposal = decision.notna() & valid_filing & decision.ge(filing)
        disposed_positive = valid_disposal & decision.gt(deadline)
        disposed_negative = valid_disposal & ~decision.gt(deadline)
        missing_disp = pd.Series(False, index=frame.index)
        for year, codes in code_to_missing.items():
            missing_disp |= frame["year"].eq(year) & frame["disp_name"].isin(codes)
        pending_positive = (
            valid_filing & decision.isna() & missing_disp & last_list.notna() &
            last_list.ge(filing) & last_list.gt(deadline) & last_list.le(pd.Timestamp(cap))
        )
        eligible = disposed_positive | disposed_negative | pending_positive
        for year in sorted(frame["year"].dropna().unique()):
            mask = frame["year"].eq(year) & eligible
            n = int(mask.sum())
            positives = int((mask & (disposed_positive | pending_positive)).sum())
            results.append({
                "filing_year": int(year),
                "listing_date_ceiling": cap,
                "eligible_cases": n,
                "over_3_years": positives,
                "within_3_years": int((mask & disposed_negative).sum()),
                "confirmed_pending_over_3_years": int((frame["year"].eq(year) & pending_positive).sum()),
                "delay_rate": positives / n if n else float("nan"),
            })
    output = pd.DataFrame(results)
    target = TABLES_OUT / f"label_sensitivity_state{state_code}_{year_start}_{year_end}.csv"
    output.to_csv(target, index=False)
    print(output.to_string(index=False))
    print(f"Wrote {target}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-start", type=int, default=2010)
    parser.add_argument("--year-end", type=int, default=2014)
    parser.add_argument("--state-code", default=STATE_CODE)
    args = parser.parse_args()
    run(args.year_start, args.year_end, args.state_code)
