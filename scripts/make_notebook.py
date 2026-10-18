"""Builds notebooks/00_sanity_check.ipynb."""
import nbformat as nbf
from pathlib import Path

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = [
md("""# 00 Sanity check and first look

Week 1 goal. Load every dataset, plot each one, and check a few numbers against a public chart (e.g. the Shanghai premium against metalcharts.org/shanghai, GLD tonnes against spdrgoldshares.com). Then build the weekly panel and run the first rolling regression."""),
code("""import sys; sys.path.insert(0, "../src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from golddemand import data, features as ft, models as md"""),
md("## 1. Macro from FRED"),
code("""macro = data.fred(["DFII10", "DTWEXBGS", "T10YIE", "GVZCLS"], start="2010-01-01")
macro.plot(subplots=True, figsize=(9, 7)); macro.tail()"""),
md("## 2. Prices from Yahoo (gold futures, GLD, USD/CNY)"),
code("""px = data.yf_close(start="2010-01-01")
px.plot(subplots=True, figsize=(9, 6)); px.tail()"""),
md("""## 3. Shanghai premium

SGE Au99.99 (CNY/g) converted to USD/oz vs the previous US close. Compare your chart with metalcharts.org/shanghai over the same dates. Large spikes on single days are usually data errors or holiday misalignment, so look at them one by one."""),
code("""sge = data.sge_au9999()
prem = ft.shanghai_premium(sge["close"], px.usdcny, px.gold_fut, lag_spot=1)
prem.premium_pct.plot(title="Shanghai gold premium, %"); prem.premium_pct.describe()"""),
md("## 4. CFTC positioning (managed money, by AVAILABLE date)"),
code("""cot = data.cot_gold(start_year=2010)
cotf = ft.cot_features(cot)
cotf.mm_net_pct_oi.plot(title="Managed money net long, % of open interest"); cotf.tail()"""),
md("## 5. GLD holdings (ETF flows)"),
code("""gld = data.gld_holdings()
gld.gld_tonnes.plot(title="GLD gold in trust, tonnes"); gld.tail()"""),
md("""## 6. Weekly panel

Everything resampled to Friday, using only what was known by Friday."""),
code("""wk = pd.concat({
    "gold": ft.to_weekly(px.gold_fut),
    "real_yield": ft.to_weekly(macro.DFII10),
    "usd": ft.to_weekly(macro.DTWEXBGS),
    "gvz": ft.to_weekly(macro.GVZCLS),
    "sh_prem": ft.to_weekly(prem.premium_pct),
    "mm_net": ft.to_weekly(cotf.mm_net_pct_oi),
    "gld_t": ft.to_weekly(gld.gld_tonnes),
}, axis=1).dropna()
wk["gold_ret"] = np.log(wk.gold).diff()
wk["d_real_yield"] = wk.real_yield.diff()
wk["d_usd"] = np.log(wk.usd).diff()
wk["d_sh_prem"] = wk.sh_prem.diff()
wk["gld_flow"] = wk.gld_t.diff()
wk = wk.dropna(); wk.tail()"""),
md("## 7. Rolling betas. When did gold stop caring about real yields?"),
code("""b = md.rolling_betas(wk.gold_ret, wk[["d_real_yield", "d_usd"]], window=104)
b[["d_real_yield", "d_usd"]].plot(subplots=True, figsize=(9, 5), title="Rolling 2y betas")"""),
md("""## 8. First predictive test

Does last week's change in the Shanghai premium predict next week's gold return? Newey-West t-stats. Run the same test on a column of random numbers to see what "nothing" looks like."""),
code("""fwd1 = wk.gold_ret.shift(-1)
for name in ["d_sh_prem", "gld_flow", "mm_net"]:
    print(name, md.predictive_regression(wk[name], fwd1, horizon=1))
rng = np.random.default_rng(0)
print("random", md.predictive_regression(pd.Series(rng.normal(size=len(wk)), wk.index), fwd1, horizon=1))"""),
md("""## Questions to answer before week 2

1. Does your Shanghai premium match public charts? What happens with `lag_spot=0`?
2. Are COT signals indexed by availability date? What changes if you (wrongly) use report dates?
3. In which year does the real-yield beta break down? What else changed then?
4. Which demand signals are just the gold price restated? Check each one's correlation with *same-week* gold returns before trusting it as a predictor.
5. How many signals and horizons have you tried so far? Keep a log. With 20 tries one will look significant by luck."""),
]
nb = nbf.v4.new_notebook(); nb["cells"] = cells
Path(__file__).resolve().parents[1].joinpath("notebooks", "00_sanity_check.ipynb").write_text(nbf.writes(nb))
