# Gold Demand vs Consensus

Starter repo for the student research project on what drives gold now, whether demand signals predict returns, and whether bank forecasts add anything. The full project guide (motivation, steps, red flags) is in the PDF your mentor sent.

**Question.** Gold's link to real yields broke around 2022, and the story since has been about who is buying. Can we measure that demand, does it predict returns or vol out of sample, and do bank price targets beat a naive forecast?

## Setup

```bash
git clone https://github.com/jennyjin0621/gold-demand-vs-consensus.git && cd gold-demand-vs-consensus
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                          # passes on synthetic data
python scripts/demo_synthetic.py   # pipeline on data with planted structure
jupyter notebook notebooks/00_sanity_check.ipynb
```

All data is free. No API keys needed except (optionally) an LLM key for Step 3.

## Layout

```
src/golddemand/
  data.py        loaders. FRED, Yahoo, SGE (akshare), CFTC COT, GLD holdings, WGC
  features.py    Shanghai premium, COT net positioning by availability date, weekly resampling, z-scores
  models.py      rolling betas, predictive regressions (Newey-West), OOS split, backtest with costs, forecast scoring
  consensus.py   LLM prompt + validator to build a bank-target dataset from news
  synthetic.py   fake weekly gold market with a planted regime break and planted signal
notebooks/00_sanity_check.ipynb   week 1, start here
scripts/demo_synthetic.py
tests/
```

## Where to go

- [ ] **Week 1 (everyone).** Run the sanity notebook. Every series plotted and spot-checked against a public source, weekly panel built, rough date of the real-yield break noted.
- **After that, it's open.** Directions we think are worth chasing (pick, combine, or pitch your own):
  - What replaced real yields? Rolling factor model with demand factors, model-implied fair value.
  - Do demand signals predict anything? Returns or GVZ, out of sample, with a random-signal control.
  - Are the banks any good? LLM-extracted bank targets scored against a naive forecast.
  - Make it tradeable. Simple rule, backtest with costs, out of sample only.
- Bring back a notebook that runs from a fresh clone and a one-page trade view grounded in what you found.

## Rules of thumb

- Index every series by when it was **available**, not when it was measured.
- A signal that is mostly a restatement of the gold price is not a demand signal. Check same-week correlation first.
- Fewer factors, longer samples. If you need 6 factors and 3 lags to get t = 2, you have found noise.
- Every result is reported with the number of things you tried to get it.
