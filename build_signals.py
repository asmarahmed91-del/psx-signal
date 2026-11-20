#!/usr/bin/env python3
"""
build_signals.py — run by GitHub Actions on a schedule.

Fetches PSX data for every ticker in watchlist.txt, computes the rules-based
technical signal for each, and writes signals.json. The web page reads that
file from the same site (no third-party relay, no CORS), so it always loads.

Rules-based technical signals, not financial advice. Never places orders.
"""

import datetime as dt
import json
import sys
import time

import numpy as np
import pandas as pd
import requests

EOD_URL = "https://dps.psx.com.pk/timeseries/eod/{sym}"
COMPANY_URL = "https://dps.psx.com.pk/company/{sym}"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://dps.psx.com.pk/",
}
TICK = 0.01
rtick = lambda x: round(round(x / TICK) * TICK, 2)


def fetch_eod(sym, s):
    r = s.get(EOD_URL.format(sym=sym), headers=HEADERS, timeout=30)
    r.raise_for_status()
    rows = r.json().get("data", [])
    if not rows:
        raise ValueError("no history")
    recs = [(row[0], float(row[1]), float(row[2]) if len(row) > 2 and row[2] is not None else 0.0)
            for row in rows]
    df = pd.DataFrame(recs, columns=["ts", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="s")
    return df.drop(columns="ts").set_index("date").sort_index()


def fetch_quote(sym, s):
    try:
        import re
        html = s.get(COMPANY_URL.format(sym=sym), headers={"User-Agent": HEADERS["User-Agent"]}, timeout=30).text
        m = re.search(r'quote__close[^>]*>\s*Rs\.?\s*([\d,]+\.?\d*)', html) or re.search(r'Rs\.?\s*([\d,]+\.\d{2})', html)
        return float(m.group(1).replace(",", "")) if m else None
    except Exception:
        return None


def rsi_wilder(c, n=14):
    d = c.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + ag / al.replace(0, np.nan))).iloc[-1]


def swing(c, lb=60, o=5):
    w = c.tail(lb).values
    last = c.iloc[-1]
    lows = [w[i] for i in range(o, len(w) - o) if w[i] == w[i - o:i + o + 1].min()]
    highs = [w[i] for i in range(o, len(w) - o) if w[i] == w[i - o:i + o + 1].max()]
    sup = max([x for x in lows if x < last], default=float(c.tail(lb).min()))
    res = min([x for x in highs if x > last], default=float(c.tail(lb).max()))
    return float(sup), float(res)


def analyse(sym, df, live):
    c = df["close"]
    price = float(live) if live else float(c.iloc[-1])
    s50 = c.rolling(50).mean().iloc[-1]
    s200 = c.rolling(200).mean().iloc[-1]
    rsi = rsi_wilder(c)
    atr = c.diff().abs().ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    v20 = df["volume"].rolling(20).mean().iloc[-1]
    vr = (df["volume"].iloc[-1] / v20) if v20 else np.nan
    sup, res = swing(c)
    hi52, lo52 = float(c.tail(252).max()), float(c.tail(252).min())
    trend = "Up" if (not np.isnan(s200) and price > s200 and s50 > s200) else \
            ("Down" if (not np.isnan(s200) and price < s200 and s50 < s200) else "Sideways")
    cands = [x for x in (sup, s50) if not np.isnan(x) and x <= price]
    buy = min(max(cands) if cands else price, price)
    near = (price - buy) / price <= 0.03 if buy else False
    stop = min(buy - 1.5 * atr, sup * 0.99)
    t1 = res
    risk = buy - stop
    t2 = buy + 2 * risk
    if not price > hi52:
        t2 = min(t2, hi52)
    rr = (t1 - buy) / risk if risk > 0 else float("nan")
    broke = price < sup and vr and vr > 1.5
    sig = "AVOID" if (trend == "Down" or broke) else \
          ("BUY" if (trend in ("Up", "Sideways") and 30 <= rsi <= 50 and near and rr >= 2) else "WATCH")
    return {"symbol": sym, "price": round(price, 2), "signal": sig, "trend": trend,
            "rsi": None if np.isnan(rsi) else round(rsi, 1), "buy": rtick(buy), "stop": rtick(stop),
            "t1": rtick(t1), "t2": rtick(t2), "rr": None if np.isnan(rr) else round(rr, 2),
            "support": rtick(sup), "hi52": rtick(hi52), "lo52": rtick(lo52),
            "asof": df.index[-1].date().isoformat()}


def main():
    try:
        with open("watchlist.txt") as f:
            tickers = [ln.strip().upper() for ln in f if ln.strip() and not ln.startswith("#")]
    except FileNotFoundError:
        tickers = ["OGDC", "HBL", "LUCK", "ENGRO", "PPL", "MCB"]

    s = requests.Session()
    items = []
    for sym in tickers:
        try:
            df = fetch_eod(sym, s)
            live = fetch_quote(sym, s)
            items.append(analyse(sym, df, live))
            print(f"OK   {sym}: {items[-1]['signal']} @ {items[-1]['price']}")
        except Exception as e:
            items.append({"symbol": sym, "signal": "ERROR", "error": str(e)})
            print(f"FAIL {sym}: {e}", file=sys.stderr)
        time.sleep(0.4)

    order = {"BUY": 0, "WATCH": 1, "AVOID": 2, "ERROR": 3}
    items.sort(key=lambda x: order.get(x["signal"], 3))
    out = {"updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "count": len(items), "items": items}
    with open("signals.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"Wrote signals.json ({len(items)} tickers)")


if __name__ == "__main__":
    main()
