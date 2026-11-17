"""Shared feature-format helpers."""
import pandas as pd


def code_token(value, prefix):
    if pd.isna(value) or str(value).strip() == "":
        return None
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return f"{prefix}{value}"
