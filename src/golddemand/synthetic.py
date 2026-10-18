"""Synthetic weekly gold market with KNOWN structure, for tests and the demo.

- Gold has a real-yield beta that switches off halfway through (a regime break)
- Last week's change in the Shanghai premium predicts this week's return (gamma)
- Everything else is noise
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_weekly(n: int = 520, beta_ry_before: float = -0.08, beta_ry_after: float = 0.0,
                beta_usd: float = -0.5, gamma: float = 0.004, noise: float = 0.018, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-02", periods=n, freq="W-FRI")
    d_ry = rng.normal(0, 0.10, n)          # weekly change in real yield, pp
    d_usd = rng.normal(0, 0.008, n)        # weekly dollar log change
    prem_chg = rng.normal(0, 1.0, n)       # weekly change in Shanghai premium, standardized
    beta_ry = np.where(np.arange(n) < n // 2, beta_ry_before, beta_ry_after)
    signal_lag = np.r_[0, prem_chg[:-1]]
    r = beta_ry * d_ry + beta_usd * d_usd + gamma * signal_lag + rng.normal(0, noise, n)
    df = pd.DataFrame({"gold_ret": r, "d_real_yield": d_ry, "d_usd": d_usd,
                       "prem_chg": prem_chg, "noise_signal": rng.normal(0, 1, n)}, index=idx)
    df["gold"] = 1200 * np.exp(df.gold_ret.cumsum())
    truth = {"break_at": idx[n // 2], "beta_ry_before": beta_ry_before,
             "beta_ry_after": beta_ry_after, "gamma": gamma}
    return df, truth
