# SN123 (MANTIS) Live Miner — Deploy Guide

A lightweight miner: every ~60s it builds 12-challenge prediction embeddings from
free data (Binance-mirror prices + order-flow, Hyperliquid funding), encrypts them as
a V2 timelock payload, and uploads to Cloudflare R2. Validators read your R2 URL from
an on-chain commitment, decrypt after maturation, and score you by salience. No GPU,
no inference server — runs on a $5/mo CPU VPS.

> **Honest expectation** (from our backtests): the bundled signals are weak-but-real
> (funding mean-reversion + contrarian order-flow). Realistic placement is mid-pack /
> tail of the ~250 earners (~$2–4/day), not top-20. Expect ~zero for the first 10 days (immunity), then your share comes off zero and climbs over the following weeks IF the signal is novel/orthogonal. The point of running is to get the
> ground-truth on-chain answer that a backtest can't give (it can't see competitors).

## Why a Linux VPS (not macOS)

The `timelock` package (Drand IBE) only ships Linux-x86 wheels (`manylinux_2_34_x86_64`);
on macOS it needs a Rust source build. A cheap Linux x86 VPS is also the right home for
an always-on 60s loop. Everything else was developed/tested on macOS.

## 1. Provision

- Any small Linux x86_64 VPS (1 vCPU / 1GB RAM is plenty), Ubuntu 22.04.
- Confirm it can reach `data-api.binance.vision` and `api.hyperliquid.xyz`. If it can
  also reach `api.binance.com`, set `BINANCE_API_URL=https://api.binance.com` in `.env`.

## 2. Install

```bash
git clone https://github.com/Barbariandev/MANTIS.git && cd MANTIS
python3.10 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install timelock requests cryptography boto3 python-dotenv numpy pandas bittensor bittensor-cli
# timelock wheels are Linux-x86 only — installs cleanly here.
# bittensor = SDK (wallet + commit); bittensor-cli = the `btcli` command used for registration.
```

## 3. Register the hotkey on SN123 (~0.30 TAO)

```bash
btcli subnet register --netuid 123 --network finney --wallet.name default --wallet.hotkey default
```

## 4. Cloudflare R2 (free tier)

1. Create a bucket (any name).
2. **Settings → Public access → enable r2.dev subdomain** → note the URL
   `https://pub-xxxx.r2.dev` → put in `R2_PUBLIC_BASE`.
3. **Manage API Tokens → Create** (Object Read & Write) → put Account ID / Access Key
   ID / Secret into `.env`.
4. Object key MUST equal your hotkey ss58 (the miner does this automatically).

## 5. Configure

```bash
cp live_miner/.env.example .env
# edit .env: HOTKEY_SS58, wallet, R2_* fields
```

## 6. One-time on-chain commit

```bash
python live_miner/run_miner.py commit
# commits  {R2_PUBLIC_BASE}/{HOTKEY_SS58}  on netuid 123
```

## 7. Run the loop (pm2 or systemd)

```bash
# quick sanity (no encrypt/upload):
python live_miner/run_miner.py --dry-run --once

# live, under pm2:
pm2 start ".venv/bin/python" --name sn123-miner -- live_miner/run_miner.py
pm2 logs sn123-miner
```

## 8. Measure (the real test)

After ~10 days (young-UID immunity clears — admin-confirmed 72,000 blocks from first non-zero submission), check your emission on taostats
`/subnets/123/metagraph` (sort by emission) or via `btcli wallet overview`. Nonzero &
rising = your signal earns salience. Flat-zero across all challenges = cut at feature
selection → the free signals aren't enough; needs proprietary alt-data.

## What to improve next (raising your salience share)

- Add CoinGlass OI/liquidation features (paid, $29/mo) to FUNDING-XSEC + XSEC.
- Replace the weak LBFGS/breakout heuristics with trained models (the backtest harness
  in `mantis_model_iteration_tool/sn123_lab/` measures OOS skill before you ship).
- Edit `live_miner/signals.py` — each challenge's signal is one isolated function.
