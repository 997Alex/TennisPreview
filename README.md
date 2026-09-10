# TennisPreview

Tennis Match Analysis and Value Betting System for ATP/WTA tennis matches.

## Features

- **Multi-source data ingestion**: Historical data (Jeff Sackmann, tennis-data.co.uk), live odds (TheOddsAPI, API-Football), news (RSS feeds), statistics (Flashscore)
- **Tiered data quality**: Automatic classification of matches by data availability (Tier 1-4) with different modeling approaches
- **Ensemble models**: Bradley-Terry, Glicko-2, Elo+Form, XGBoost, LightGBM with per-surface calibration
- **Value betting**: Kelly criterion with fractional sizing, tier-aware thresholds
- **Risk management**: Daily/weekly exposure limits, stop-loss, correlation controls
- **Paper trading**: Full bet logging, performance tracking, daily reports

## Data Quality Tiers

| Tier | Tournaments | Min Edge | Kelly | Approach |
|------|-------------|----------|-------|----------|
| 1 | GS, Masters 1000, ATP/WTA 250-500 | 2.5% | 25% | Full ensemble (BT, Glicko, XGB, Market) |
| 2 | Challengers 50-175, WTA 125 | 3.5% | 18% | Reduced ensemble |
| 3 | Challengers 25, ITF W100/75 | 5.0% | 10% | Market-heavy blend |
| 4 | ITF W50/W15, Qualies, Junior | - | 0% | Monitor only, no betting |

## Installation

```bash
# Clone and enter directory
cd TennisPreview

# Run setup script (creates venv, installs deps)
./run.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Edit `.env` with your API keys:

```bash
# Required (free tiers available)
THEODDSAPI_KEY=your_key        # https://the-odds-api.com/
API_FOOTBALL_KEY=your_key      # https://api-sports.io/

# Optional
BETFAIR_APP_KEY=your_key       # https://developer.betfair.com/
```

## Usage

```bash
# Run for today (live matches, real odds)
./run.sh

# Run for specific date
./run.sh --date 2024-01-15

# Run for next 7 days
./run.sh --days 7

# Update historical data only
./run.sh --update-data

# Offline sandbox (simulated odds, clearly labeled)
./run.sh --demo

# Disable progress bars
./run.sh --no-progress
```

> The program outputs **decisions only** (who is value + why): no paper money,
> no stakes, it never bets automatically.

## Output

- **Console**: Real-time progress bars, predictions table, final decisions with
  reasons (why each side is favored), Sisal coverage flag
- **Logs**: `logs/tennispreview_YYYY-MM-DD.log`
- **Pipeline reports**: `output/pipeline_report_YYYY-MM-DD.json`
  (analyzed events + decisions with reasons, no money involved)

## Example Output

```
============================================================
RESULTS FOR 2024-01-15
============================================================

Match                                    Player                 Odds   Edge   Stake Tier
-----------------------------------------------------------------------
Alcaraz vs Medvedev                      Alcaraz               1.85  4.2%   125.00    1
Djokovic vs Sinner                       Sinner                2.10  3.8%   110.00    1
Swiatek vs Sabalenka                     Swiatek               1.75  5.1%   140.00    1

RISK SUMMARY
------------------------------------------------------------
Bankroll:            10000.00
Daily Exposure:       3.75%
Weekly Exposure:      8.20%
Daily PnL:              0.00
Bets Today:               3
```

## Architecture

```
src/
├── data/ingest/         # Data sources (Sackmann, OddsAPI, Flashscore, News)
├── data/merge/          # Odds merging & deduplication
├── sports/tennis/
│   ├── features/        # Feature engineering with tiering
│   └── models/          # Baseline (BT, Glicko, Elo) + ML (XGB, LGBM)
├── betting/             # Value detection, risk engine, paper trading
├── pipeline/            # Daily orchestration
└── utils/               # Config, logging, models
```
## License

MIT License - See LICENSE file for details.