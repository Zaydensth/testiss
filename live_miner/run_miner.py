"""MANTIS (SN123) live miner loop.

Every ~60s: refresh feeds -> build 12-challenge embeddings -> encrypt as V2 payload
-> write file named exactly your hotkey -> upload to Cloudflare R2 (overwrite).
Once: commit the R2 object URL on-chain.

Commands:
  python live_miner/run_miner.py --dry-run --once    # build+validate, no encrypt/upload (works on macOS)
  python live_miner/run_miner.py commit              # one-time: commit R2 URL on-chain (needs bittensor)
  python live_miner/run_miner.py                     # live loop (VPS: needs `timelock`)

Env (.env in repo root, see live_miner/.env.example):
  HOTKEY_SS58, BT_WALLET_NAME, BT_WALLET_HOTKEY, BT_CHAIN_ENDPOINT,
  R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, R2_PUBLIC_BASE,
  LOCK_SECONDS (default 30), CYCLE_SECONDS (default 60)
"""

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root (config.py, generate_and_encrypt.py)
sys.path.insert(0, _HERE)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(_HERE), ".env"))
except Exception:
    pass

import config                                  # noqa: E402
from feeds import PriceFeed, FundingFeed       # noqa: E402
from signals import build_embeddings           # noqa: E402


def _asset_universe():
    price_assets = {"ETH", "BTC"}
    funding_assets = set()
    for c in config.CHALLENGES:
        a = c.get("assets")
        if a:
            price_assets.update(a)
        if c["loss_func"] == "funding_xsec":
            funding_assets.update(a or [])
    return sorted(price_assets), sorted(funding_assets)


def make_feeds():
    pa, fa = _asset_universe()
    print(f"[feeds] price assets={len(pa)} funding assets={len(fa)}", flush=True)
    price = PriceFeed(pa, maxlen=6000)
    print("[feeds] warming price (one-time, may take a few minutes)...", flush=True)
    price.warmup(bars=4200)
    funding = FundingFeed(fa)
    print("[feeds] warming funding (Hyperliquid)...", flush=True)
    funding.warmup(days=45)
    print(f"[feeds] price.ok={len(price.ok)}/{len(pa)} funding.ok={len(funding.ok)}/{len(fa)}", flush=True)
    return price, funding


def build_payload(hotkey, embeddings, lock_seconds):
    """Encrypt V2 payload. Imports timelock lazily (only available on the VPS)."""
    from generate_and_encrypt import generate_v2
    return generate_v2(hotkey=hotkey, lock_seconds=lock_seconds,
                       owner_pk_hex=config.OWNER_HPKE_PUBLIC_KEY_HEX,
                       payload_text=None, embeddings=embeddings)


def upload_r2(hotkey, payload_dict):
    import boto3
    from botocore.config import Config as BotoConfig
    acct = os.environ["R2_ACCOUNT_ID"]
    s3 = boto3.client(
        "s3", endpoint_url=f"https://{acct}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=BotoConfig(signature_version="s3v4"), region_name="auto")
    body = json.dumps(payload_dict, separators=(",", ":")).encode()
    s3.put_object(Bucket=os.environ["R2_BUCKET"], Key=hotkey, Body=body,
                  ContentType="application/json")
    return len(body)


def cmd_commit():
    """One-time: commit the R2 object URL on-chain so validators can find your payload."""
    import bittensor as bt
    base = os.environ["R2_PUBLIC_BASE"].rstrip("/")   # e.g. https://pub-xxxx.r2.dev
    hotkey = os.environ["HOTKEY_SS58"]
    url = f"{base}/{hotkey}"
    wallet = bt.wallet(name=os.environ["BT_WALLET_NAME"], hotkey=os.environ["BT_WALLET_HOTKEY"])
    sub = bt.subtensor(network=os.environ.get("BT_CHAIN_ENDPOINT", "finney"))
    print(f"[commit] committing {url} on netuid {config.NETUID} ...", flush=True)
    sub.commit(wallet, config.NETUID, url)
    print("[commit] done. Validators will read your R2 URL from chain.", flush=True)


def run(dry_run=False, once=False):
    hotkey = os.environ.get("HOTKEY_SS58", "DRYRUN_HOTKEY")
    lock_s = int(os.environ.get("LOCK_SECONDS", "30"))
    cycle_s = int(os.environ.get("CYCLE_SECONDS", "60"))
    price, funding = make_feeds()
    n = 0
    while True:
        t0 = time.time()
        if n > 0:
            price.update(); funding.update()
        emb = build_embeddings(price, funding)
        if dry_run:
            obj = {c["ticker"]: v for v, c in zip(emb, config.CHALLENGES)}
            obj["hotkey"] = hotkey
            print(f"[dry] cycle {n}: built {len(emb)} challenge embeddings; "
                  f"ETHLBFGS argmax={max(range(5), key=lambda i: emb[2][i])} "
                  f"ETH={[round(x,3) for x in emb[0]]} "
                  f"plaintext_bytes={len(json.dumps(obj,separators=(',',':')))}", flush=True)
        else:
            payload = build_payload(hotkey, emb, lock_s)
            nbytes = upload_r2(hotkey, payload)
            print(f"[live] cycle {n}: round={payload['round']} uploaded {nbytes}B to R2 key={hotkey}", flush=True)
        if once:
            break
        dt = time.time() - t0
        time.sleep(max(1.0, cycle_s - dt))
        n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="run", choices=["run", "commit"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    if args.command == "commit":
        cmd_commit()
    else:
        run(dry_run=args.dry_run, once=args.once)


if __name__ == "__main__":
    main()
