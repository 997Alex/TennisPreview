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

### Windows

```bat
cd C:\Users\Ale\Desktop\TennisPreview
python -m venv venv
venv\Scripts\pip.exe install -r requirements.txt
venv\Scripts\python.exe -m playwright install chromium
copy .env.example .env
run.bat --demo --no-progress   :: verifica installazione
run.bat --no-progress          :: palinsesto del giorno (live)
run.bat --gui                  :: interfaccia grafica analisi singolo match
```

## Fonti dati live (gratis, senza chiave)

- **Match del giorno**: ESPN scoreboard API (ATP/WTA, live + programmati)
- **Quote/mercato**: Polymarket (Gamma public API, crowd senza vig)
- **News per giocatore**: RSS ATP/WTA + Google News, con scoring
  infortuni/malattie/ritiri (`infortunio`, `withdraw`, `illness`...)
  e forma recente (`titolo`, `recuperato`...)
- **Storico H2H/forma**: risultati ESPN recenti rigiocati su profili
  Elo di superficie (in attesa del DB storico Sackmann in `data/sackmann/`)
- **Sisal.it**: palinsesto+quote del giorno via Playwright (`SISAL_HEADLESS=0`
  per browser visibile). Sisal usa anti-bot aggressivo: se lo scraping
  fallisce, importa il palinsesto a mano:

```bat
:: palinsesto.txt: una riga per match "A;B;quota1;quota2;Torneo"
venv\Scripts\python.exe scripts\import_sisal.py palinsesto.txt
run.bat --no-progress
```

Le quote Sisal importate vengono unite ai match ESPN per nome e usate
come mercato primario nel report.

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

# Launch GUI for single match analysis
./run.sh --gui
```

### GUI Mode

The GUI provides a graphical interface for single-match analysis:
1. Enter player names (Giocatore 1 vs Giocatore 2)
2. Click **Analizza Match**
3. View results in two tabs:
   - **Risultati**: Win probability, odds, Polymarket sentiment, Sisal coverage
   - **Dettaglio**: Full analysis including H2H, news impact, surface Elo, form, serve/return ratings, and value bet reasoning

```bat
:: Windows GUI mode
run.bat --gui
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