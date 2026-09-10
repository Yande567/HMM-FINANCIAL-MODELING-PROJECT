# Detecting Financial Market Regimes with Hidden Markov Models

**ZCAS University — MSc Data Science · AI & Machine Learning · Group 11**

> **Research question:** How can we automatically detect changes in financial
> market regimes from historical observations?
>
> **Objective:** Identify whether the S&P 500 is in a Bull, Bear, or
> high-volatility state, using an unsupervised Gaussian Hidden Markov Model.

---

## 1. Approach in one paragraph

The market regime is never directly observed — no data feed announces "we are
 in a bear market." What *is* observable is the daily behaviour of prices.
We therefore model the regime as the **hidden state** of a Hidden Markov Model,
and the daily `[log return, rolling volatility]` pair as the **observation**
emitted by that state. Baum-Welch (EM) learns the transition and emission
parameters from unlabelled data; the Viterbi algorithm then decodes the single
most likely regime sequence over the full five-year history.

The reason an HMM is used rather than clustering (k-means, GMM) is the
**transition matrix**. Clustering labels each day independently and produces a
chart that flickers between regimes day to day. Viterbi maximises the
probability of the whole *path*, so the transition term penalises switching and
one violent day inside a bull market does not flip the label.

---

## 2. Repository structure

```
ZCAS-G11-PROJECT/
├── README.md                  <- you are here
├── requirements.txt           <- direct dependencies, with reasons
├── requirements.lock.txt      <- exact resolved versions (generated, see §4)
├── .gitignore
│
├── src/                       <- all reusable pipeline code
│   ├── __init__.py
│   ├── config.py              <- every constant that shapes a result
│   ├── data_loader.py         <- Task 1.2  fetch + cache ^GSPC
│   ├── preprocessing.py       <- Tasks 1.3-1.5  clean, returns, volatility
│   ├── features.py            <- Task 1.6  build the (T, 2) matrix
│   └── model.py               <- Tasks 3.2-3.5  fit, decode, label
│
├── notebooks/                 <- exploration and demos
│   └── demo_price_vs_returns.py
│
├── data/
│   ├── raw/                   <- unmodified download
│   └── processed/             <- derived features (the model's input)
│
├── models/                    <- serialised trained model (joblib)
├── reports/figures/           <- figures for the slide deck
├── tests/                     <- pytest suite
└── app.py                     <- Phase 4 Streamlit dashboard
```

**Why `raw/` and `processed/` are separate.** The raw download is treated as
immutable evidence: if a result looks wrong, we can always rebuild from it and
see exactly which transformation introduced the problem. Overwriting the raw
file with cleaned data destroys that audit trail — and, since yfinance scrapes
an unofficial endpoint that periodically breaks, it may not be re-downloadable
on the day before submission.

---

## 3. Why we commit `data/` and `models/`

Conventional advice is to gitignore data. We deliberately do not, for three
project-specific reasons:

1. **Size.** Five years of daily bars is roughly 1,255 rows — about 100 KB.
   This is not a big-data problem.
2. **Group reproducibility.** Person A trains the model; Person B loads it into
   Streamlit. If the model file is not in the repo, Person B must retrain, and
   any divergence in library versions or seeds produces a different set of
   regimes than the ones on the slides.
3. **Marker reproducibility.** yfinance is an unofficial scraper. If it is
   broken or rate-limited on marking day, a repo without committed data cannot
   be run at all.

*Caveat:* Yahoo Finance data is licensed for personal use. Keep this repository
private, and do not redistribute the raw file outside the group and the marker.

---

## 4. Environment setup

Requires **Python 3.11 or newer**.

```bash
# from the project root
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Then freeze the exact resolved environment and commit it:

```bash
pip freeze > requirements.lock.txt
```

**Why two files.** `requirements.txt` states *intent* — the handful of packages
we actually chose, with version bounds and the reasoning behind them; it is
meant to be read. `requirements.lock.txt` states *fact* — every package in the
tree pinned to the exact version that produced our results; it is meant to be
executed. Only the second guarantees a marker reproduces our numbers; only the
first explains why we chose what we chose.

Verify the install:

```bash
python -c "import hmmlearn, yfinance, streamlit; print('environment OK')"
```

---

## 5. Sprint plan and ownership

| Phase | Scope | Owner | Days |
|---|---|---|---|
| 1 | Data management and processing | **Person A** | 1–3 |
| 2 | Mathematical foundations and EDA | Person B | 4–5 |
| 3 | HMM training, decoding, regime labelling | **Person A** | 6–8 |
| 4 | Streamlit dashboard | Person B | 9–11 |
| 5 | Slide deck and final review | Both | 12–14 |

Branch per phase, e.g. `feature/phase1-data-pipeline`, merged into `master`
after review by the other member.

---

## 6. Method notes

Detailed derivations, assumptions and their limitations, the price-vs-returns
control experiment, and the label-switching and look-ahead-bias analyses are
maintained as living notes alongside this repository (Note 01 — Concept
Foundations).

---

## 7. References

- Russell, S. & Norvig, P. (2021). *Artificial Intelligence: A Modern
  Approach* (4th ed.). Pearson.
- Rabiner, L. (1989). A Tutorial on Hidden Markov Models and Selected
  Applications in Speech Recognition. *Proceedings of the IEEE*, 77(2),
  257–286.
- Bishop, C. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Hamilton, J. D. (1989). A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle. *Econometrica*, 57(2),
  357–384. — the original regime-switching model for financial time series.
