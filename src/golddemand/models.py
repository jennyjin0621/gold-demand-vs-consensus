"""Models and tests. Deliberately simple, the hard part is honesty, not the model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS


def rolling_betas(y: pd.Series, X: pd.DataFrame, window: int = 104) -> pd.DataFrame:
    """Rolling OLS betas of y on X (weekly data, default 2y window).
    Use it to see WHEN relationships change, e.g. gold vs real yields after 2022."""
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    res = RollingOLS(df.y, sm.add_constant(df[X.columns]), window=window).fit()
    return res.params


def predictive_regression(signal: pd.Series, fwd_ret: pd.Series, horizon: int) -> dict:
    """fwd_ret(t, t+h) = a + b * signal(t). Newey-West SEs with lag = horizon,
    because overlapping forward returns make plain OLS t-stats far too big."""
    df = pd.concat([signal.rename("x"), fwd_ret.rename("y")], axis=1).dropna()
    res = sm.OLS(df.y, sm.add_constant(df.x)).fit(cov_type="HAC", cov_kwds={"maxlags": max(horizon, 1)})
    return {"beta": res.params["x"], "t_nw": res.tvalues["x"], "r2": res.rsquared, "n": int(res.nobs)}


def oos_split(df: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick the split date BEFORE looking at results and write it down."""
    return df.loc[:split], df.loc[split:].iloc[1:]


def backtest(position: pd.Series, ret: pd.Series, cost_bps: float = 2.0,
             periods_per_year: int = 52) -> dict:
    """Position decided at t earns the return from t to t+1. Costs on turnover.

    position: target exposure in [-1, 1] known at the END of period t
    ret:      simple return of the asset from t to t+1, aligned at t+1
    """
    pos = position.reindex(ret.index).ffill().fillna(0)
    held = pos.shift(1).fillna(0)                  # the lag that prevents look-ahead
    turnover = pos.diff().abs().fillna(pos.abs())
    pnl = held * ret - turnover * cost_bps / 1e4
    pnl = pnl.dropna()
    ann = np.sqrt(periods_per_year)
    return {
        "sharpe": float(pnl.mean() / pnl.std() * ann) if pnl.std() > 0 else 0.0,
        "ann_return": float(pnl.mean() * periods_per_year),
        "hit_rate": float((pnl[held.loc[pnl.index] != 0] > 0).mean()),
        "turnover_per_year": float(turnover.mean() * periods_per_year),
        "pnl": pnl,
    }


def score_forecasts(forecasts: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    """Score price targets against what actually happened and against a naive forecast.

    forecasts needs columns published_date, target_date, target_usd_oz, bank.
    Naive forecast = price on the publish date (random walk). A futures-implied
    forecast is better still if you have the right contract, so try that next.
    """
    p = price.sort_index()

    def px(d):
        s = p.loc[:pd.Timestamp(d)]
        return s.iloc[-1] if len(s) else np.nan

    f = forecasts.copy()
    f = f[pd.to_datetime(f.target_date) <= p.index[-1]]
    f["spot_at_publish"] = f.published_date.map(px)
    f["realized"] = f.target_date.map(px)
    f["err_bank"] = np.log(f.target_usd_oz / f.realized)
    f["err_naive"] = np.log(f.spot_at_publish / f.realized)
    f["bank_beats_naive"] = f.err_bank.abs() < f.err_naive.abs()
    f["direction_right"] = np.sign(f.target_usd_oz - f.spot_at_publish) == np.sign(f.realized - f.spot_at_publish)
    return f
