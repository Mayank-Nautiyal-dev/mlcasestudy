"""Project paths and first-run cohort settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasetddl" / "csv"
CASE_DIR = DATA_ROOT / "cases" / "cases"
ACTS_FILE = DATA_ROOT / "acts_sections.csv"
KEYS_DIR = DATA_ROOT / "keys" / "keys"

RESULT_DIR = ROOT / "result"
DATA_OUT = RESULT_DIR / "data"
TABLES_OUT = RESULT_DIR / "tables"
FIGURES_OUT = RESULT_DIR / "figures"
MODELS_OUT = RESULT_DIR / "models"
CONCLUSION_DIR = ROOT / "conclusion"

# First reproducible temporal study; these are configuration defaults, not a final
# claim that this geography/cohort is representative of Indian courts.
FILING_YEAR = 2014
STATE_CODE = "01"  # Maharashtra in the supplied state/court lookup files.
RANDOM_SEED = 2026
# Used only to reject implausible far-future listing dates when confirming a
# pending case survived beyond its three-year anniversary; this is not treated
# as the source extraction cutoff.
MAX_PLAUSIBLE_LISTING_DATE = "2021-12-31"

for directory in (DATA_OUT, TABLES_OUT, FIGURES_OUT, MODELS_OUT, CONCLUSION_DIR):
    directory.mkdir(parents=True, exist_ok=True)
