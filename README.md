# Gold Demand vs Consensus

Starter repo for the student research project on what drives gold now, whether demand signals predict returns, and whether bank forecasts add anything. The full project guide (motivation, steps, red flags) is in the PDF your mentor sent.

**Question.** Gold's link to real yields broke around 2022, and the story since has been about who is buying. Can we measure that demand, does it predict returns or vol out of sample, and do bank price targets beat a naive forecast?

## Setup

```bash
git clone <this repo> && cd gold-demand-vs-consensus
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

## Week-by-week checklist

- [ ] **Week 1.** Run the sanity notebook. Every series plotted and spot-checked against a public source.
- [ ] **Week 2.** Factor model. Rolling betas on real yields and dollar, find and date the break. Add demand factors (Shanghai premium, GLD flows, COT, central bank buying).
- [ ] **Week 3.** Signal tests. Pick an out-of-sample split date first and write it down. Predictive regressions at 1w and 4w for returns and for GVZ changes. Keep a log of every test you run.
- [ ] **Week 4.** Consensus dataset. Extract 50+ bank targets with the LLM pipeline, hand-check 20%, score vs naive.
- [ ] **Week 5.** Strategy. Simple rule from surviving signals, backtest with costs, out of sample only.
- [ ] **Week 6.** Trade memo. Where does the model disagree with consensus right now? Options expression, size, invalidation level.

## Rules of thumb

- Index every series by when it was **available**, not when it was measured.
- A signal that is mostly a restatement of the gold price is not a demand signal. Check same-week correlation first.
- Fewer factors, longer samples. If you need 6 factors and 3 lags to get t = 2, you have found noise.
- Every result is reported with the number of things you tried to get it.
