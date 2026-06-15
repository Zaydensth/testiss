"""Live data feeds for the MANTIS miner: Binance-mirror 1m klines (price + order-flow)
and Hyperliquid hourly funding. Rolling in-memory buffers — warm once, then append
the newest bars each cycle (cheap enough for a ~60s loop over ~50 assets).

Geo notes (same as the backtest harness): api.binance.com is geoblocked from some
regions; the public data mirror data-api.binance.vision works. OKX/Bybit are
geoblocked too — Hyperliquid's public /info endpoint is the free funding source.
Override BINANCE_API_URL if your VPS can reach the main API.
"""

import os
import time

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BINANCE_API = os.environ.get("BINANCE_API_URL", "https://data-api.binance.vision").rstrip("/")
HL_URL = "https://api.hyperliquid.xyz/info"
HL_COIN = {"SHIB": "kSHIB", "PEPE": "kPEPE", "BONK": "kBONK", "FLOKI": "kFLOKI"}


def _session():
    s = requests.Session()
    r = Retry(total=5, connect=5, read=5, status=5, backoff_factor=1.0,
              status_forcelist=(429, 500, 502, 503, 504), allowed_methods=["GET", "POST"],
              raise_on_status=False)
    a = HTTPAdapter(max_retries=r, pool_connections=8, pool_maxsize=16)
    s.mount("https://", a)
    return s


_S = _session()


def _ticker_symbol(ticker):
    return f"{ticker}USDT"


class PriceFeed:
    """Rolling 1m OHLCV + order-flow buffers for a set of crypto tickers."""

    def __init__(self, tickers, maxlen=6000):
        self.tickers = list(tickers)
        self.maxlen = maxlen
        self.close = {t: np.array([], dtype=np.float64) for t in self.tickers}
        self.ofi = {t: np.array([], dtype=np.float64) for t in self.tickers}   # taker imbalance [-1,1]
        self.tstamp = {t: np.array([], dtype=np.int64) for t in self.tickers}  # bar open_time ms
        self.ok = set()

    def _fetch(self, symbol, start_ms=None, limit=1000):
        params = {"symbol": symbol, "interval": "1m", "limit": limit}
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        try:
            r = _S.get(f"{BINANCE_API}/api/v3/klines", params=params, timeout=20)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    def _append(self, t, rows):
        if not rows:
            return
        c = np.array([float(x[4]) for x in rows])
        vol = np.array([float(x[5]) for x in rows])
        tbb = np.array([float(x[9]) for x in rows])
        ts = np.array([int(x[0]) for x in rows], dtype=np.int64)
        of = np.where(vol > 0, 2.0 * np.clip(tbb / np.where(vol > 0, vol, 1), 0, 1) - 1.0, 0.0)
        # drop overlap with existing tail
        if len(self.tstamp[t]) and len(ts):
            keep = ts > self.tstamp[t][-1]
            c, of, ts = c[keep], of[keep], ts[keep]
        self.close[t] = np.concatenate([self.close[t], c])[-self.maxlen:]
        self.ofi[t] = np.concatenate([self.ofi[t], of])[-self.maxlen:]
        self.tstamp[t] = np.concatenate([self.tstamp[t], ts])[-self.maxlen:]

    def warmup(self, bars=4000):
        now = int(time.time() * 1000)
        for t in self.tickers:
            sym = _ticker_symbol(t)
            start = now - bars * 60_000
            got_any = False
            s = start
            while True:
                rows = self._fetch(sym, start_ms=s, limit=1000)
                if not rows:
                    break
                self._append(t, rows)
                got_any = True
                s = rows[-1][0] + 60_000
                if len(rows) < 1000 or s > now:
                    break
                time.sleep(0.05)
            if got_any and len(self.close[t]) > 100:
                self.ok.add(t)

    def update(self):
        for t in self.tickers:
            sym = _ticker_symbol(t)
            start = (self.tstamp[t][-1] + 60_000) if len(self.tstamp[t]) else None
            rows = self._fetch(sym, start_ms=start, limit=10)
            self._append(t, rows)
            time.sleep(0.03)


class FundingFeed:
    """Rolling hourly funding from Hyperliquid for a set of tickers."""

    def __init__(self, tickers):
        self.tickers = list(tickers)
        self.ftime = {t: np.array([], dtype=np.int64) for t in self.tickers}
        self.frate = {t: np.array([], dtype=np.float64) for t in self.tickers}
        self.ok = set()

    def _fetch(self, coin, start_ms):
        try:
            r = _S.post(HL_URL, json={"type": "fundingHistory", "coin": coin,
                                      "startTime": start_ms, "endTime": int(time.time() * 1000)}, timeout=20)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    def _append(self, t, rows):
        if not rows:
            return
        ts = np.array([int(x["time"]) for x in rows], dtype=np.int64)
        rt = np.array([float(x["fundingRate"]) for x in rows], dtype=np.float64)
        if len(self.ftime[t]) and len(ts):
            keep = ts > self.ftime[t][-1]
            ts, rt = ts[keep], rt[keep]
        self.ftime[t] = np.concatenate([self.ftime[t], ts])
        self.frate[t] = np.concatenate([self.frate[t], rt])

    def warmup(self, days=45):
        now = int(time.time() * 1000)
        for t in self.tickers:
            coin = HL_COIN.get(t, t)
            cur = now - days * 86_400_000
            while cur < now:
                rows = self._fetch(coin, cur)
                if not rows:
                    break
                self._append(t, rows)
                last = rows[-1]["time"]
                if last <= cur or len(rows) < 2:
                    break
                cur = last + 1
                time.sleep(0.1)
            if len(self.frate[t]) > 24:
                self.ok.add(t)

    def update(self):
        for t in self.tickers:
            coin = HL_COIN.get(t, t)
            start = (self.ftime[t][-1] + 1) if len(self.ftime[t]) else int(time.time() * 1000) - 7200_000
            self._append(t, self._fetch(coin, start))
            time.sleep(0.05)

    def latest(self, t):
        a = self.frate.get(t)
        return float(a[-1]) if a is not None and len(a) else np.nan

    def change(self, t, hours):
        a = self.frate.get(t)
        if a is None or len(a) <= hours:
            return np.nan
        return float(a[-1] - a[-1 - hours])


class ValidatorFeed:
    """Polls the validator's own latest_prices.json (config.PRICE_DATA_URL) — the
    EXACT prices + funding the validator labels/scores against (Polygon spot +
    Bybit perp funding). Using this aligns our features with the labels, which is
    the single biggest free signal-quality lever. Keeps a rolling per-minute
    funding history (persisted to disk so restarts don't lose it) so we can
    compute the 8h funding-change the FUNDING-XSEC label is built from.

    Funding here is the source the validator actually uses (Bybit), unlike
    Hyperliquid — so it matches the label, not just a correlated proxy.
    """

    def __init__(self, url, cache_dir=None, maxlen=4320):  # 4320 min = 3 days
        self.url = url
        self.maxlen = maxlen
        self.ts = []                 # epoch-sec of each poll
        self.funding = {}            # asset -> list[rate]
        self.price = {}              # asset -> latest price
        root = cache_dir or os.environ.get("MANTIS_DATA_DIR", os.path.join(os.path.dirname(__file__), ".cache"))
        os.makedirs(root, exist_ok=True)
        self.path = os.path.join(root, "validator_funding.jsonl")
        self._load()

    def _load(self):
        import json
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                for line in f.readlines()[-self.maxlen:]:
                    rec = json.loads(line)
                    self.ts.append(rec["t"])
                    for a, v in rec["f"].items():
                        self.funding.setdefault(a, []).append(float(v))
        except Exception:
            pass

    def poll(self):
        import json
        try:
            r = _S.get(self.url, timeout=15)
            if r.status_code != 200:
                return False
            d = r.json()
        except Exception:
            return False
        pr = d.get("prices", {}) or {}
        fr = d.get("funding_rates", {}) or {}
        self.price = {a: float(v) for a, v in pr.items() if isinstance(v, (int, float))}
        # only append a new funding row if the snapshot timestamp advanced
        t = d.get("timestamp") or len(self.ts)
        if self.ts and self.ts[-1] == t:
            return True
        self.ts.append(t)
        for a, v in fr.items():
            if isinstance(v, (int, float)):
                self.funding.setdefault(a, []).append(float(v))
        # trim
        if len(self.ts) > self.maxlen:
            self.ts = self.ts[-self.maxlen:]
            for a in self.funding:
                self.funding[a] = self.funding[a][-self.maxlen:]
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps({"t": t, "f": fr}) + "\n")
        except Exception:
            pass
        return True

    def change(self, asset, minutes):
        """Funding-rate change over the last `minutes` of polled history."""
        a = self.funding.get(asset)
        if a is None or len(a) <= minutes:
            return np.nan
        return float(a[-1] - a[-1 - minutes])

    def get_price(self, asset):
        return self.price.get(asset, np.nan)

    @property
    def n(self):
        return len(self.ts)
