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

# Named artefacts, so no string literals leak into the pipeline modules.
RAW_PRICES_CSV = RAW_DIR / "gspc_raw.csv"              # Task 1.2 output
PROCESSED_CSV = PROCESSED_DIR / "observations.csv"     # Task 1.6 output
MODEL_FILE = MODELS_DIR / "gaussian_hmm_3state.joblib"  # Task 4.2 input

# ===========================================================================
# DATA ACQUISITION  (Sprint Task 1.2)
# ===========================================================================
TICKER = "^GSPC"  # S&P 500 index. The caret prefix marks a Yahoo index rather
                  # than a tradeable ticker. "SPY" would be the ETF — similar
                  # but not identical, since an ETF has fees and tracking error.

START_DATE = "2021-09-01"
END_DATE = "2026-09-01"

PRICE_COLUMN = "Adj Close"

# ===========================================================================
# FEATURE ENGINEERING  (Sprint Tasks 1.4 - 1.6)
# ===========================================================================
VOLATILITY_WINDOW = 10

TRADING_DAYS_PER_YEAR = 252
ANNUALISE_VOLATILITY = False
# Kept off. In theory annualising is a pure rescaling and cannot change the
# decoded regimes; in practice it does, because hmmlearn seeds the state means
# with k-means, which is not scale-invariant. Measured: the two scaling agreed
# at only ARI 0.58-0.87. See notebooks/demo_feature_scaling.py.

FEATURE_COLUMNS = ["log_return", "volatility"]
# Order defines the observation matrix: means_[i][0] is return, [i][1] is
# volatility everywhere downstream, including the Task 3.5 labelling logic.


# ===========================================================================
# MODEL  (Sprint Tasks 3.1 - 3.5)
# ===========================================================================
N_REGIMES = 3           # Bull / Bear / third state — named from the data, not
                        # hardcoded (see Note 01 §7). Phase 3 compares
                        # n_components = 2, 3, 4 by BIC to justify this.
COVARIANCE_TYPE = "full"

N_ITER = 1000           # max Baum-Welch (EM) iterations
TOL = 1e-4              # convergence tolerance on log-likelihood improvement
RANDOM_SEED = 42

N_RESTARTS = 20
# EM only reaches a local optimum. Across five seeds on identical data, accuracy
# against known regimes ranged 0.52-0.75, so a single seed reports an accident
# of initialisation. Phase 3 keeps the highest-likelihood fit of N_RESTARTS and
# reports the spread. Note: log-likelihood (and AIC/BIC) are only comparable
# between models fitted to identically scaled features.

# --- Chronological split (Task 3.1) ---------------------------------------
TRAIN_END_DATE = "2025-09-01"
