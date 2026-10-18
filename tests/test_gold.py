import numpy as np
import pandas as pd

from golddemand import features as ft, models as md, consensus as cs
from golddemand.synthetic import make_weekly


def test_shanghai_premium_formula():
    idx = pd.date_range("2026-01-05", periods=3, freq="B")
    sge = pd.Series(1000.0, idx)              # CNY per gram
    fx = pd.Series(7.0, idx)
    spot = pd.Series(1000 / 7 * 31.1034768, idx)  # makes premium exactly 0
    out = ft.shanghai_premium(sge, fx, spot, lag_spot=0)
    assert np.allclose(out.premium_pct, 0)
    lagged = ft.shanghai_premium(sge, fx, spot, lag_spot=1)
    assert len(lagged) == 2                   # first day has no prior US close


def test_rolling_betas_find_the_break():
    df, t = make_weekly()
    b = md.rolling_betas(df.gold_ret, df[["d_real_yield", "d_usd"]], window=104)
    before = b.d_real_yield.loc[:t["break_at"]].dropna().mean()
    after = b.d_real_yield.loc[t["break_at"] + pd.Timedelta(weeks=110):].mean()
    assert before < -0.05 and abs(after) < 0.035


def test_predictive_regression_planted_vs_noise():
    df, t = make_weekly()
    fwd = df.gold_ret.shift(-1)
    real = md.predictive_regression(df.prem_chg, fwd, horizon=1)
    fake = md.predictive_regression(df.noise_signal, fwd, horizon=1)
    assert real["t_nw"] > 3 and abs(fake["t_nw"]) < 2.5


def test_backtest_lag_and_costs():
    df, _ = make_weekly()
    ret = np.exp(df.gold_ret) - 1
    good = md.backtest(np.sign(df.prem_chg), ret, cost_bps=2)
    noise = md.backtest(np.sign(df.noise_signal), ret, cost_bps=2)
    assert good["sharpe"] > 0.8 and noise["sharpe"] < 0.6
    # a look-ahead position (knowing this week's return) must NOT be what backtest does
    cheat = md.backtest(np.sign(ret.shift(-1)), ret, cost_bps=0)
    assert cheat["sharpe"] > 5      # sanity: perfect foresight is obviously huge


def test_cot_available_date_shift():
    cot = pd.DataFrame({"open_interest": [100, 100], "mm_long": [60, 70], "mm_short": [20, 20]},
                       index=pd.to_datetime(["2026-01-06", "2026-01-13"]))
    cot["available_date"] = cot.index + pd.Timedelta(days=3)
    f = ft.cot_features(cot)
    assert f.index[0] == pd.Timestamp("2026-01-09") and f.mm_net_pct_oi.iloc[1] == 50


def test_consensus_validation_rejects_hallucinations():
    text = "Goldman Sachs raised its end-2027 gold forecast to $4,900 an ounce."
    reply = '''Sure! [{"bank": "Goldman Sachs", "target_usd_oz": 4900, "target_date": "2027-12-31"},
                {"bank": "UBS", "target_usd_oz": 5200, "target_date": "2027-06-30"}]'''
    df = cs.validate(cs.parse_llm_json(reply), text, "2026-10-01")
    assert list(df.bank) == ["Goldman Sachs"]


def test_score_forecasts():
    p = pd.Series([100.0, 110.0, 120.0], index=pd.to_datetime(["2026-01-01", "2026-06-01", "2026-12-31"]))
    f = pd.DataFrame({"bank": ["A", "B"], "published_date": ["2026-01-01"] * 2,
                      "target_date": ["2026-12-31"] * 2, "target_usd_oz": [118.0, 90.0]})
    s = md.score_forecasts(f, p)
    assert list(s.bank_beats_naive) == [True, False] and list(s.direction_right) == [True, False]
