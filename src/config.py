"""
Import it as:      from src import config
                   df = pd.read_csv(config.RAW_PRICES_CSV)
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


RAW_PRICES_CSV = RAW_DIR / "gspc_raw.csv"              # Task 1.2 output
PROCESSED_CSV = PROCESSED_DIR / "observations.csv"     # Task 1.6 output
MODEL_FILE = MODELS_DIR / "gaussian_hmm_3state.joblib"  # Task 4.2 input


TICKER = "^GSPC"  # S&P 500 index.

START_DATE = "2021-09-01"
END_DATE = "2026-09-01"

PRICE_COLUMN = "Adj Close"

VOLATILITY_WINDOW = 10

TRADING_DAYS_PER_YEAR = 252
ANNUALISE_VOLATILITY = False   # moot under standardisation; a constant cancels

FEATURE_COLUMNS = ["log_return", "volatility"]

STANDARDISE_FEATURES = True
# Scaler fitted on TRAIN ONLY. Raw variances are 1.3e-4 / 2.9e-5 against
# hmmlearn defaults calibrated for scale 1: 8/20 restarts crashed on raw
# features, 0/20 standardised.

N_REGIMES = 3           # Bull / Bear / third state — named from the data,

COVARIANCE_TYPE = "diag"
# BIC 3708.8 vs 3726.6 for full, near-identical regimes. Off-diagonals do not
# earn their parameters.

N_ITER = 1000           # max Baum-Welch (EM) iterations
TOL = 1e-4              # convergence tolerance on log-likelihood improvement
RANDOM_SEED = 42

N_RESTARTS = 20

# --- Chronological split (Task 3.1) ---------------------------------------
TRAIN_END_DATE = "2025-09-01"
