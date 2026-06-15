"""Build the 12-element MANTIS embeddings list (config.CHALLENGES order) from live
feeds, using the best free signals validated in the backtest harness:

  LBFGS      : vol-forecast Gaussian bucket distribution (clustering + drift tilt)
  FUNDING-XSEC: contrarian funding-change (mean reversion)  [best free signal, +0.0265]
  XSEC-RANK  : contrarian order-flow imbalance              [+0.0225]
  MULTI-BREAKOUT: contrarian momentum (bet reversal)
  ETH binary/hitfirst: contrarian order-flow + vol
  FX/metals (CAD/NZD/CHF/XAG): neutral (no free data feed)
  TRADE-MIX  : contrarian short-term reversal signed position

All outputs clipped to the validator's accepted ranges (ledger._validate_submission).
These are weak-but-real signals — expect tail/mid-pack placement, not top-20.
"""

import numpy as np

import config

EDGES = np.array([-2.0, -1.0, 1.0, 2.0])
_FX = {"CADUSD", "NZDUSD", "CHFUSD", "XAGUSD"}


def _sig(x):
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))


def _ncdf(x, s):
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / (s * sqrt(2.0) + 1e-12)))


def _blocks_to_bars(blocks_ahead):
    # SAMPLE_EVERY=5 blocks = 1 sample(1m bar); blocks_ahead in 12s blocks
    return max(1, int(round(blocks_ahead / 5)))


def _lbfgs_vec(close, horizon_bars, alpha=1.5, mu_gain=2.0):
    lp = np.log(np.clip(close, 1e-12, None))
    if len(lp) < 3600 + horizon_bars + 10:
        # not enough history -> neutral climatology
        mu, s = 0.0, 1.0
    else:
        r_h = lp[horizon_bars:] - lp[:-horizon_bars]
        trail = r_h[-3600:].std() + 1e-12
        recent = r_h[-600:].std() + 1e-12
        drift = lp[-1] - lp[-1 - horizon_bars]
        mu = (drift / trail) * mu_gain
        s = float(np.clip((recent / trail) ** alpha, 0.3, 4.0))
    c = [_ncdf(e - mu, s) for e in EDGES]
    b = np.array([c[0], c[1] - c[0], c[2] - c[1], c[3] - c[2], 1.0 - c[3]])
    b = np.clip(b, 1e-6, 1 - 1e-6); b /= b.sum()
    q = [float(np.clip(2 * (1 - _ncdf(k, s)), 1e-6, 1 - 1e-6)) for _ in range(4) for k in (0.5, 1.0, 2.0)]
    return [float(x) for x in b] + q


def _ofi_recent(ofi, win):
    if len(ofi) < win:
        return 0.0
    return float(np.mean(ofi[-win:]))


def _mom_z(close, win):
    if len(close) < win + 2:
        return 0.0
    lp = np.log(np.clip(close[-(win + 1):], 1e-12, None))
    sd = np.diff(lp).std() * np.sqrt(win) + 1e-9
    return float((lp[-1] - lp[0]) / sd)


def _ema_reversal_pos(close, win=120, smooth=0.99, lookback=2000):
    """Converged EMA-smoothed REVERSAL position for TRADE-MIX. Recomputed each
    cycle from price history (deterministic, no cross-cycle state). Heavy
    smoothing keeps turnover ~0.004 so the validator's 20bp/|delta-pos| fee
    doesn't erase the edge — backtest: rev+smooth0.99 Sharpe ~+3 vs churn ~-5."""
    if len(close) < win + 50:
        return 0.0
    lp = np.log(np.clip(close[-lookback:], 1e-12, None))
    pos = 0.0
    for t in range(win, len(lp)):
        sd = np.diff(lp[t - win:t + 1]).std() * np.sqrt(win) + 1e-9
        z = (lp[t] - lp[t - win]) / sd
        pos = (1.0 - smooth) * (-np.tanh(z)) + smooth * pos
    return float(np.clip(pos, -1.0, 1.0))


def build_embeddings(price: "PriceFeed", funding: "FundingFeed") -> list:
    out = []
    for c in config.CHALLENGES:
        tk = c["ticker"]
        lf = c["loss_func"]

        if lf == "binary" and tk in _FX:
            out.append([0.5, 0.5])                              # neutral, no data

        elif lf == "binary":                                   # ETH directional
            pk = c.get("price_key", tk)
            of = _ofi_recent(price.ofi.get(pk, np.array([])), 60)
            p_up = _sig(-3.0 * of)                              # contrarian order-flow
            out.append([p_up, 1.0 - p_up])

        elif lf == "hitfirst":
            pk = c.get("price_key", "ETH")
            cl = price.close.get(pk, np.array([]))
            of = _ofi_recent(price.ofi.get(pk, np.array([])), 60)
            # crude: vol sets P(neither); drift/of tilts up vs down
            if len(cl) > 600:
                r = np.diff(np.log(np.clip(cl[-600:], 1e-12, None)))
                vr = (r[-60:].std() + 1e-12) / (r.std() + 1e-12)
            else:
                vr = 1.0
            p_neither = float(np.clip(0.5 / max(vr, 0.5), 0.1, 0.8))
            rem = 1.0 - p_neither
            tilt = _sig(-3.0 * of)                              # contrarian
            out.append([rem * tilt, rem * (1 - tilt), p_neither])

        elif lf == "lbfgs":
            pk = c.get("price_key", tk)
            h = _blocks_to_bars(c["blocks_ahead"])
            out.append(_lbfgs_vec(price.close.get(pk, np.array([])), h))

        elif lf == "range_breakout_multi":                     # contrarian breakout -> reversal
            d = {}
            # contrarian (reversal): strong move+order-flow -> bet reversal.
            # backtest: mom+ofi sign=- win=30 -> OOS AUC ~0.538 (clears >0.5 gate).
            # only a[0] is scored by the validator; a[1] kept valid but unused.
            for a in c["assets"]:
                cl = price.close.get(a, np.array([]))
                mom = _mom_z(cl, 30)
                ofi = _ofi_recent(price.ofi.get(a, np.array([])), 30)
                s = 0.7 * np.tanh(mom) + 0.3 * ofi
                p_cont = float(np.clip(_sig(-s), 0.01, 0.99)) if len(cl) > 32 else 0.5
                d[a] = [p_cont, round(1.0 - p_cont, 6)]
            out.append(d)

        elif lf == "xsec_rank":                                # contrarian order-flow
            raw = {a: -_ofi_recent(price.ofi.get(a, np.array([])), 480) for a in c["assets"]}
            med = np.median(list(raw.values())) if raw else 0.0
            out.append({a: float(np.clip(v - med, -1, 1)) for a, v in raw.items()})

        elif lf == "funding_xsec":                             # contrarian funding change (8h)
            raw = {a: -(funding.change(a, 8) if not np.isnan(funding.change(a, 8)) else 0.0)
                   for a in c["assets"]}
            med = np.median(list(raw.values())) if raw else 0.0
            # scale tiny funding deltas into [-1,1] by rank-ish normalization
            vals = np.array(list(raw.values()))
            sd = vals.std() + 1e-12
            out.append({a: float(np.clip((v - med) / sd, -1, 1)) for a, v in raw.items()})

        elif lf == "trade_mix":                                # persistent low-turnover reversal
            d = {}
            for a in c["assets"]:
                cl = price.close.get(a, np.array([]))
                d[a] = _ema_reversal_pos(cl, win=120, smooth=0.99)
            out.append(d)

        else:                                                  # unknown -> zeros
            out.append([0.0] * c["dim"])
    return out
