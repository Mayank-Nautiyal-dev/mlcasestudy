"""Describe observed delay patterns in the chronological test cohort."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_OUT, FIGURES_OUT, KEYS_DIR, STATE_CODE, TABLES_OUT


def grouped_rates(frame, group_column, min_cases=100):
    stats = frame.groupby(group_column, dropna=False)["delay_over_3_years"].agg(
        cases="size", delayed="sum", delay_rate="mean"
    ).reset_index()
    stats = stats.loc[stats["cases"] >= min_cases].copy()
    stats["rate_low_95"] = stats["delay_rate"] - 1.96 * np.sqrt(
        stats["delay_rate"] * (1 - stats["delay_rate"]) / stats["cases"]
    )
    stats["rate_high_95"] = stats["delay_rate"] + 1.96 * np.sqrt(
        stats["delay_rate"] * (1 - stats["delay_rate"]) / stats["cases"]
    )
    stats["rate_low_95"] = stats["rate_low_95"].clip(0, 1)
    stats["rate_high_95"] = stats["rate_high_95"].clip(0, 1)
    return stats.sort_values(["delay_rate", "cases"], ascending=[False, False])


def plot_rates(stats, label, title, path, n=20):
    if stats.empty:
        return
    shown = stats.head(n).sort_values("delay_rate")
    fig, ax = plt.subplots(figsize=(11, max(6, min(12, .32 * len(shown)))))
    ax.barh(shown[label].astype(str), shown["delay_rate"], color="#3568a8")
    ax.set_xlabel("Share observed to take more than three years")
    ax.set_title(title)
    ax.set_xlim(0, min(1, max(.1, float(shown["delay_rate"].max()) * 1.12)))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyze(year_start=2010, test_year=2014, state_code=STATE_CODE):
    state_code = str(state_code).zfill(2)
    path = DATA_OUT / f"temporal_features_state{state_code}_{year_start}_{test_year}.csv"
    usecols = ["year", "delay_over_3_years", "case_type", "court_name", "act_section_tokens"]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    frame = frame.loc[
        frame["year"].astype(str).eq(str(test_year)) & frame["delay_over_3_years"].notna()
    ].copy()
    frame["delay_over_3_years"] = frame["delay_over_3_years"].astype(int)

    for column, name, display in [
        ("case_type", "case_type", "case type"),
        ("court_name", "court", "court"),
    ]:
        table = grouped_rates(frame, column)
        table_path = TABLES_OUT / f"temporal_delay_by_{name}_state{state_code}_test{test_year}.csv"
        table.to_csv(table_path, index=False)
        plot_rates(
            table, column, f"Highest observed delay rates by {display} (n >= 100)",
            FIGURES_OUT / f"temporal_delay_by_{name}_state{state_code}_test{test_year}.png",
        )

    act_rows = frame[["delay_over_3_years", "act_section_tokens"]].copy()
    act_rows["act_section_token"] = act_rows["act_section_tokens"].fillna("").str.split()
    act_rows = act_rows.explode("act_section_token").dropna(subset=["act_section_token"])
    act_rates = grouped_rates(act_rows, "act_section_token")

    def normalized_id(value):
        if pd.isna(value):
            return None
        text = str(value).strip()
        return text[:-2] if text.endswith(".0") else text

    acts = pd.read_csv(KEYS_DIR / "act_key.csv", dtype=str, low_memory=False)
    sections = pd.read_csv(KEYS_DIR / "section_key.csv", dtype=str, low_memory=False)
    act_labels = {
        normalized_id(row.act): row.act_s for row in acts.itertuples(index=False)
        if normalized_id(row.act) is not None and pd.notna(row.act_s)
    }
    section_labels = {
        normalized_id(row.section): row.section_s for row in sections.itertuples(index=False)
        if normalized_id(row.section) is not None and pd.notna(row.section_s)
    }
    act_rates["act_id"] = act_rates["act_section_token"].str.extract(r"A([^_]+)", expand=False)
    act_rates["section_id"] = act_rates["act_section_token"].str.extract(r"S([^_]+)", expand=False)
    act_rates["act_label"] = act_rates["act_id"].map(act_labels)
    act_rates["section_label"] = act_rates["section_id"].map(section_labels)
    act_rates["chart_label"] = (
        act_rates["act_section_token"].astype(str) + " | " +
        act_rates["act_label"].fillna("unknown act") + " / " +
        act_rates["section_label"].fillna("unknown section")
    ).str.slice(0, 95)
    act_rates.to_csv(
        TABLES_OUT / f"temporal_delay_by_act_section_state{state_code}_test{test_year}.csv", index=False
    )
    plot_rates(
        act_rates, "chart_label", "Highest observed delay rates by act-section (n >= 100)",
        FIGURES_OUT / f"temporal_delay_by_act_section_state{state_code}_test{test_year}.png",
    )
    print(f"Temporal test-cohort pattern tables and figures written for {len(frame):,} labeled cases.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-start", type=int, default=2010)
    parser.add_argument("--test-year", type=int, default=2014)
    parser.add_argument("--state-code", default=STATE_CODE)
    args = parser.parse_args()
    analyze(args.year_start, args.test_year, args.state_code)
