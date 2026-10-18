"""Turn raw series into model-ready weekly features.

Two timing rules that matter more than any model choice
1. Use data only from when it was AVAILABLE. COT is as of Tuesday, released Friday.
   WGC central bank stats come out with a lag of 1-2 months.
2. Shanghai closes ~15:30 Beijing time, before New York even opens. Comparing the
   SGE close with the same-date US close mixes two different moments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TROY_OZ_GRAMS = 31.1034768


def shanghai_premium(sge_cny_per_g: pd.Series, usdcny: pd.Series, spot_usd_oz: pd.Series,
                     lag_spot: int = 1) -> pd.DataFrame:
    """Premium of SGE gold over the Western price, in %.

    sge_usd_oz = SGE (CNY/g) / USDCNY * 31.1035
    premium    = sge_usd_oz / spot - 1

    lag_spot=1 compares the SGE close on day t with the US close on day t-1, which
    is the last Western price known when Shanghai closes. Try 0 and see how much
    noisier the series gets.
    """
    df = pd.concat({"sge": sge_cny_per_g, "usdcny": usdcny,
                    "spot": spot_usd_oz.shift(lag_spot)}, axis=1).dropna()
    df["sge_usd_oz"] = df.sge / df.usdcny * TROY_OZ_GRAMS
    df["premium_pct"] = (df.sge_usd_oz / df.spot - 1) * 100
    return df


def cot_features(cot: pd.DataFrame) -> pd.DataFrame:
    """Managed money net position as % of open interest, indexed by AVAILABLE date."""
    out = pd.DataFrame(index=pd.DatetimeIndex(cot["available_date"], name="date"))
    out["mm_net_pct_oi"] = ((cot.mm_long - cot.mm_short) / cot.open_interest * 100).values
    out["mm_net_chg"] = out.mm_net_pct_oi.diff()
    return out


def to_weekly(daily: pd.DataFrame, how: str = "last", rule: str = "W-FRI") -> pd.DataFrame:
    """Resample to weekly (Friday) using only data observed by that Friday."""
    return getattr(daily.sort_index().resample(rule), how)()


def zscore(x: pd.Series, window: int = 52, min_periods: int = 26) -> pd.Series:
    """Rolling z-score that uses only the past (no full-sample mean)."""
    m = x.rolling(window, min_periods=min_periods).mean()
    s = x.rolling(window, min_periods=min_periods).std()
    return (x - m) / s


def log_returns(price: pd.Series) -> pd.Series:
    return np.log(price).diff()


def forward_returns(price: pd.Series, h: int) -> pd.Series:
    """Log return from t to t+h, aligned at t. This is the TARGET, never a feature."""
    return np.log(price.shift(-h) / price)
