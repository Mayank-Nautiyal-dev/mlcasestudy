"""Audit date coverage for a selected filing-year/state without inferring a cutoff."""
import argparse

import pandas as pd

from config import CASE_DIR, STATE_CODE, TABLES_OUT


DATE_FIELDS = [
    "date_of_filing", "date_of_decision", "date_last_list", "date_next_list",
]


def audit(year=2018, state_code=STATE_CODE):
    state_code = str(state_code).zfill(2)
    source = CASE_DIR / f"cases_{year}.csv"
    chunks = []
    total_rows = 0
    state_rows = 0
    for part in pd.read_csv(
        source, usecols=["state_code", *DATE_FIELDS], dtype=str,
        chunksize=250_000, low_memory=False,
    ):
        total_rows += len(part)
        code = part["state_code"].str.replace(r"\.0$", "", regex=True).str.zfill(2)
        part = part.loc[code.eq(state_code)].copy()
        state_rows += len(part)
        if len(part):
            chunks.append(part)
    frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=DATE_FIELDS)
    parsed = {}
    for col in DATE_FIELDS:
        raw = frame[col].astype("string").str.strip()
        parsed[col] = pd.to_datetime(raw.mask(raw.str.startswith("5000-", na=False)), errors="coerce")

    rows = [
        ("filing_year_source_rows", total_rows),
        ("selected_state_rows", state_rows),
        ("missing_or_sentinel_decision_date", int(parsed["date_of_decision"].isna().sum())),
        ("valid_decision_dates", int(parsed["date_of_decision"].notna().sum())),
        ("valid_decisions_after_2018_12_31", int(parsed["date_of_decision"].gt(pd.Timestamp("2018-12-31")).sum())),
    ]
    for col in ["date_of_decision", "date_last_list", "date_next_list"]:
        series = parsed[col].dropna()
        if not series.empty:
            rows.append((f"{col}_minimum_valid", series.min().date().isoformat()))
            rows.append((f"{col}_maximum_valid", series.max().date().isoformat()))
            rows.append((f"{col}_valid_after_2018_12_31", int(series.gt(pd.Timestamp("2018-12-31")).sum())))

    output = pd.DataFrame(rows, columns=["measure", "value"])
    path = TABLES_OUT / f"date_coverage_state{state_code}_year{year}.csv"
    output.to_csv(path, index=False)
    print(output.to_string(index=False))
    print("Date maxima describe recorded case fields; they do not establish the source snapshot cutoff.")
    print(f"Wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--state-code", default=STATE_CODE)
    args = parser.parse_args()
    audit(args.year, args.state_code)
