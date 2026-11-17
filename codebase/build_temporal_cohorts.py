"""Build and combine labelable cohorts for a chronological experiment."""
import argparse

import pandas as pd

from build_cohort import build
from config import DATA_OUT, STATE_CODE, TABLES_OUT


def run(years=(2010, 2011, 2012, 2013, 2014), state_code=STATE_CODE):
    state_code = str(state_code).zfill(2)
    paths = [build(year, state_code) for year in years]
    cohort = pd.concat(
        [pd.read_csv(path, dtype={"ddl_case_id": str}, low_memory=False) for path in paths],
        ignore_index=True,
    )
    output = DATA_OUT / f"temporal_cohort_state{state_code}_{min(years)}_{max(years)}.csv"
    cohort.to_csv(output, index=False)
    flow_frames = []
    for year in years:
        path = TABLES_OUT / f"sample_flow_state{state_code}_year{year}.csv"
        flow = pd.read_csv(path)
        flow.insert(0, "filing_year", year)
        flow_frames.append(flow)
    flow_path = TABLES_OUT / f"temporal_sample_flow_state{state_code}_{min(years)}_{max(years)}.csv"
    pd.concat(flow_frames, ignore_index=True).to_csv(flow_path, index=False)
    prevalence = (
        cohort.dropna(subset=["delay_over_3_years"])
        .groupby("year")["delay_over_3_years"]
        .agg(cases="size", delayed="sum", delay_rate="mean")
        .reset_index()
    )
    prevalence.to_csv(
        TABLES_OUT / f"delay_prevalence_by_year_state{state_code}_{min(years)}_{max(years)}.csv",
        index=False,
    )
    print(f"Combined cohort written: {output}")
    print(prevalence.to_string(index=False))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2010, 2011, 2012, 2013, 2014])
    parser.add_argument("--state-code", default=STATE_CODE)
    args = parser.parse_args()
    run(tuple(args.years), args.state_code)
