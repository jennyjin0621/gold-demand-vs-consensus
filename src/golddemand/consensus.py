"""Build a dataset of bank gold price targets from public news, using an LLM.

Workflow
1. Collect article text (bank research summaries reported by Reuters, Bloomberg,
   Kitco, CNBC, bank press pages, etc). Save each as a .txt in data/raw/articles/
   with the URL on the first line.
2. Run `extract_targets(text, llm)` on each one. `llm` is any function that takes
   a prompt string and returns the model's text reply, so use whatever model
   you have access to.
3. Hand-check at least 20% of the rows against the article. LLMs invent numbers.
4. Save to data/consensus_targets.csv and score with models.score_forecasts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import pandas as pd

COLUMNS = ["published_date", "bank", "target_usd_oz", "target_date", "horizon_label",
           "scenario", "reason", "source_url", "quote"]

PROMPT = """You extract gold price forecasts from financial news.

Return a JSON list. One object per distinct forecast in the article, with keys
  bank            institution making the forecast (e.g. "Goldman Sachs")
  target_usd_oz   number, USD per troy ounce
  target_date     ISO date the forecast refers to. "end-2026" -> "2026-12-31",
                  "Q2 2027" -> "2027-06-30", "12 months" -> publish date + 1 year
  horizon_label   the horizon in the article's own words
  scenario        "base", "bull" or "bear"
  reason          the main driver the bank gives, at most 12 words
  quote           the exact sentence containing the number

Only include numbers that are explicitly forecasts of the gold price. Ignore
silver, ignore past prices. If there are none, return [].

Article published {published_date}:
---
{text}
---
JSON:"""


def parse_llm_json(reply: str) -> list[dict]:
    """Pull the first JSON list out of a model reply (models like to add prose)."""
    m = re.search(r"\[.*\]", reply, flags=re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)]


def validate(rows: list[dict], text: str, published_date: str, source_url: str = "") -> pd.DataFrame:
    """Keep rows whose number actually appears in the article and whose date is sane."""
    out = []
    digits = re.sub(r"[,\s]", "", text)
    for r in rows:
        try:
            tgt = float(str(r.get("target_usd_oz", "")).replace(",", "").replace("$", ""))
            tdate = pd.Timestamp(r.get("target_date"))
        except (ValueError, TypeError):
            continue
        if not 300 < tgt < 50_000:
            continue
        if str(int(tgt)) not in digits:          # hallucination guard
            continue
        if tdate < pd.Timestamp(published_date):
            continue
        out.append({**{c: r.get(c, "") for c in COLUMNS}, "target_usd_oz": tgt,
                    "target_date": tdate.date().isoformat(), "published_date": published_date,
                    "source_url": source_url})
    return pd.DataFrame(out, columns=COLUMNS)


def extract_targets(text: str, llm: Callable[[str], str], published_date: str,
                    source_url: str = "") -> pd.DataFrame:
    reply = llm(PROMPT.format(text=text[:20_000], published_date=published_date))
    return validate(parse_llm_json(reply), text, published_date, source_url)


def anthropic_llm(model: str = "claude-sonnet-4-5") -> Callable[[str], str]:
    """Example `llm` using the Anthropic SDK (needs ANTHROPIC_API_KEY). Swap the
    model name for whatever is current, or write the same 5 lines for any other
    provider or a local model."""
    import anthropic
    client = anthropic.Anthropic()

    def call(prompt: str) -> str:
        msg = client.messages.create(model=model, max_tokens=2000,
                                     messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    return call


def template(path: str | Path = "data/consensus_targets.csv") -> None:
    """Write an empty CSV with the right columns, for manual entry."""
    pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)
