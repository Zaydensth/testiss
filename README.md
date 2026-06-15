# SN123 (MANTIS) Live Miner

Lightweight miner for Bittensor Subnet 123 (MANTIS — "The Ultimate Signal Machine").
Every ~60s it builds prediction embeddings for all 12 challenges from **free data**
(Binance-mirror prices + order-flow, Hyperliquid funding), encrypts them as a V2
timelock payload, and uploads to Cloudflare R2. Validators read your R2 URL from an
on-chain commitment, decrypt after maturation, and score you by salience.

**No GPU, no inference server, no Chutes bill** — runs on a ~$5/mo Linux x86 VPS
(1 vCPU/1GB is enough; 4 vCPU is overkill).

## Signals (free, weak-but-real)

- **LBFGS** (ETH/BTC): volatility-regime forecast (vol clustering + drift tilt)
- **FUNDING-XSEC**: contrarian funding mean-reversion (strongest free signal in backtest)
- **XSEC-RANK**: contrarian order-flow imbalance
- **MULTI-BREAKOUT**: contrarian momentum (bet reversal)
- **ETH binary/hitfirst**: contrarian order-flow + volatility
- **FX/metals** (CAD/NZD/CHF/XAG): neutral (no free feed)

## Honest expectation

Backtested OOS, these signals sit just below the solo viability gate. Realistic
on-chain placement is **mid-pack / tail of the ~250 earners (~$2–4/day)**, not top-20.
Running it is the only way to get the ground-truth competitive answer (a backtest
can't see other miners' encrypted submissions). To climb: add CoinGlass OI/liquidation
data and train real models for the LBFGS/breakout challenges.

## Quick start

See **[live_miner/DEPLOY.md](live_miner/DEPLOY.md)**. In short, on a Linux x86 VPS:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
btcli subnet register --netuid 123 --network finney   # ~0.30 TAO
cp live_miner/.env.example .env   # fill HOTKEY_SS58, wallet, R2_* (Cloudflare R2 free tier)
python live_miner/run_miner.py commit                 # one-time: commit R2 URL on-chain
pm2 start ".venv/bin/python" --name sn123-miner -- live_miner/run_miner.py
```

Validate without secrets first: `python live_miner/run_miner.py --dry-run --once`

## Credits

`config.py` and `generate_and_encrypt.py` are from the MANTIS subnet
([github.com/Barbariandev/MANTIS](https://github.com/Barbariandev/MANTIS), MIT) and
define the on-chain payload format. The `live_miner/` code is the signal + submission loop.
