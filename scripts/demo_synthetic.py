"""Run the modelling pipeline on synthetic data with planted structure.

    python scripts/demo_synthetic.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from golddemand import models as md
from golddemand.synthetic import make_weekly

df, truth = make_weekly()
betas = md.rolling_betas(df.gold_ret, df[["d_real_yield", "d_usd"]], window=104)
fwd = df.gold_ret.shift(-1)
real = md.predictive_regression(df.prem_chg, fwd, horizon=1)
fake = md.predictive_regression(df.noise_signal, fwd, horizon=1)
train, test = md.oos_split(df, "2019-12-31")
ret = np.exp(test.gold_ret) - 1
bt_real = md.backtest(np.sign(test.prem_chg), ret, cost_bps=2)
bt_fake = md.backtest(np.sign(test.noise_signal), ret, cost_bps=2)

print(f"planted real-yield beta {truth['beta_ry_before']} -> {truth['beta_ry_after']} at {truth['break_at'].date()}")
print(f"premium signal   t(NW) = {real['t_nw']:.2f}  beta = {real['beta']:.4f} (truth {truth['gamma']})")
print(f"noise signal     t(NW) = {fake['t_nw']:.2f}")
print(f"OOS Sharpe after 2bp costs, real {bt_real['sharpe']:.2f}, noise {bt_fake['sharpe']:.2f}")

fig, ax = plt.subplots(2, 1, figsize=(9, 6))
ax[0].plot(betas.index, betas.d_real_yield)
ax[0].axvline(truth["break_at"], color="k", ls="--", lw=1)
ax[0].set_title("Rolling 2y beta of gold to real yields (synthetic, break at dashed line)")
ax[1].plot(bt_real["pnl"].cumsum(), label="premium signal")
ax[1].plot(bt_fake["pnl"].cumsum(), label="noise signal")
ax[1].set_title("Out-of-sample cumulative PnL after costs")
ax[1].legend()
fig.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "demo_synthetic.png", dpi=120)
print("saved figures/demo_synthetic.png")
