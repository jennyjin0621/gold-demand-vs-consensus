"""Free data loaders for the gold project. No API keys needed.

Source                      What                                   Loader
FRED (fredgraph.csv)        real yields, dollar, breakevens, GVZ   fred()
Yahoo Finance (yfinance)    gold futures, GLD, USD/CNY             yf_close()
Shanghai Gold Exchange      Au99.99 benchmark, CNY per gram        sge_au9999()  (via akshare)
CFTC public reporting       Commitments of Traders, COMEX gold     cot_gold()
SPDR Gold Shares            GLD tonnes of gold held, daily         gld_holdings()
World Gold Council          central bank buying (free signup)      wgc_table()   (manual download)

Downloads are cached in data/cache/. Delete the cache to refresh.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0 (student research project)"}

# Useful FRED series
FRED_SERIES = {
    "DFII10": "10y TIPS real yield, %",
    "DGS10": "10y nominal Treasury yield, %",
    "T10YIE": "10y breakeven inflation, %",
    "DTWEXBGS": "Broad trade-weighted US dollar index",
    "GVZCLS": "CBOE gold ETF volatility index (GVZ)",
    "VIXCLS": "VIX",
}

GOLD_COT_CODE = "088691"  # COMEX gold (100 oz)
TROY_OZ_GRAMS = 31.1034768


def _cached(name: str, fetch, use_cache: bool = True) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.csv"
    if use_cache and path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = fetch()
    df.to_csv(path)
    return df


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


# --------------------------------------------------------------------------
def fred(series: list[str] | str, start: str = "2006-01-01", use_cache: bool = True) -> pd.DataFrame:
    series = [series] if isinstance(series, str) else list(series)

    def fetch():
        out = []
        for s in series:
            r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                             params={"id": s}, headers=UA, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), na_values=".")
            df = df.rename(columns={df.columns[0]: "date"}).set_index("date")
            df.index = pd.to_datetime(df.index)
            out.append(pd.to_numeric(df[s], errors="coerce"))
        return pd.concat(out, axis=1)

    df = _cached(f"fred_{'_'.join(series)}", fetch, use_cache)
    return df.loc[start:]


def yf_close(tickers: dict[str, str] | None = None, start: str = "2006-01-01",
             use_cache: bool = True) -> pd.DataFrame:
    """Daily closes. Default tickers: front gold future, GLD, onshore USD/CNY."""
    tickers = tickers or {"gold_fut": "GC=F", "gld": "GLD", "usdcny": "CNY=X"}

    def fetch():
        import yfinance as yf
        out = {}
        for name, t in tickers.items():
            df = yf.download(t, start=start, auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            out[name] = df["Close"]
        return pd.DataFrame(out)

    return _cached(f"yf_{'_'.join(tickers.values())}_{start}", fetch, use_cache)


def sge_au9999(use_cache: bool = True) -> pd.DataFrame:
    """SGE Au99.99 daily prices in CNY per gram.

    Uses the open-source `akshare` package (pip install akshare), which reads the
    SGE website. If that breaks, download the history by hand from sge.com.cn
    into data/raw/sge_au9999.csv with columns date, close.
    """
    manual = RAW / "sge_au9999.csv"

    def fetch():
        if manual.exists():
            df = pd.read_csv(manual)
        else:
            import akshare as ak
            df = ak.spot_hist_sge(symbol="Au99.99")
        df.columns = [_norm(c) for c in df.columns]
        df = df.rename(columns={"time": "date"})
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()

    return _cached("sge_au9999", fetch, use_cache)


def cot_gold(start_year: int = 2010, use_cache: bool = True) -> pd.DataFrame:
    """Disaggregated COT (futures only) for COMEX gold.

    Positions are as of Tuesday but published the following Friday afternoon.
    Use `available_date` (not the report date) when building signals, or you are
    trading on information you did not have yet.
    """
    def fetch():
        try:
            url = "https://publicreporting.cftc.gov/resource/72hh-3qpy.csv"
            q = {"$where": f"cftc_contract_market_code='{GOLD_COT_CODE}'",
                 "$limit": 50000, "$order": "report_date_as_yyyy_mm_dd"}
            r = requests.get(url, params=q, headers=UA, timeout=60)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
        except Exception:
            # fallback: yearly zip files from cftc.gov
            frames = []
            for y in range(start_year, pd.Timestamp.today().year + 1):
                r = requests.get(f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{y}.zip",
                                 headers=UA, timeout=60)
                r.raise_for_status()
                z = zipfile.ZipFile(io.BytesIO(r.content))
                frames.append(pd.read_csv(z.open(z.namelist()[0]), low_memory=False))
            df = pd.concat(frames)
        df.columns = [_norm(c) for c in df.columns]
        code_col = next(c for c in df.columns if c.startswith("cftc_contract_market_code"))
        df = df[df[code_col].astype(str).str.zfill(6) == GOLD_COT_CODE]
        date_col = next(c for c in df.columns if c.startswith("report_date_as_yyyy_mm_dd"))
        keep = {
            date_col: "report_date",
            "open_interest_all": "open_interest",
            "m_money_positions_long_all": "mm_long",
            "m_money_positions_short_all": "mm_short",
            "prod_merc_positions_long_all": "producer_long",
            "prod_merc_positions_short_all": "producer_short",
            "swap_positions_long_all": "swap_long",
            "swap_positions_short_all": "swap_short",
        }
        df = df[[c for c in keep if c in df.columns]].rename(columns=keep)
        df["report_date"] = pd.to_datetime(df["report_date"])
        df = df.drop_duplicates("report_date").set_index("report_date").sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")
        df["available_date"] = df.index + pd.Timedelta(days=3)  # Tue -> Fri release
        return df

    df = _cached("cot_gold", fetch, use_cache)
    df["available_date"] = pd.to_datetime(df["available_date"])
    return df.loc[str(start_year):]


def gld_holdings(use_cache: bool = True) -> pd.DataFrame:
    """Daily tonnes of gold held by GLD (the biggest gold ETF).

    SPDR publishes a free archive CSV. If the URL moves, download it from
    spdrgoldshares.com (Historical Archive) into data/raw/gld_archive.csv.
    """
    manual = RAW / "gld_archive.csv"

    def fetch():
        if manual.exists():
            text = manual.read_text(errors="ignore")
        else:
            r = requests.get("https://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.csv",
                             headers=UA, timeout=60)
            r.raise_for_status()
            text = r.text
        lines = text.splitlines()
        hdr = next(i for i, l in enumerate(lines) if "date" in l.lower() and "tonnes" in l.lower())
        df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
        df.columns = [_norm(c) for c in df.columns]
        tcol = next(c for c in df.columns if "tonnes" in c)
        out = pd.DataFrame({"date": pd.to_datetime(df["date"], errors="coerce", format="mixed"),
                            "gld_tonnes": pd.to_numeric(df[tcol].astype(str).str.replace(",", ""), errors="coerce")})
        return out.dropna().set_index("date").sort_index()

    return _cached("gld_holdings", fetch, use_cache)


def wgc_table(path: str | Path, sheet: str | int = 0, header_row: int | None = None) -> pd.DataFrame:
    """Read a World Gold Council Goldhub download (free account needed).

    Goldhub spreadsheets have a few title rows on top. Open the file once, note
    the header row, and pass it in. Central bank stats live under
    gold.org/goldhub/data/monthly-central-bank-statistics.
    """
    df = pd.read_excel(path, sheet_name=sheet, header=header_row)
    df.columns = [_norm(c) for c in df.columns]
    return df
