#!/usr/bin/env python3
"""One-time backfill of recent tour results (ESPN week scoreboards).

Fills data/recent/espn_results.json from 2026-06-01 to the given end date
(default: today), so profiles price current form even though the Sackmann
snapshot stops in May 2026. Afterwards every bot startup appends only the
current week automatically.

Usage:
  python3 scripts/backfill_espn.py
  python3 scripts/backfill_espn.py --start 2026-06-01 --end 2026-09-07
"""
import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.daily import TennisDailyPipeline
from src.utils.logging import setup_logging


def mondays(start: date, end: date):
    d = start - timedelta(days=start.weekday())
    while d <= end:
        yield d
        d += timedelta(days=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default=date.today().isoformat())
    a = ap.parse_args()
    setup_logging(log_level="INFO")

    start = date.fromisoformat(a.start)
    end = date.fromisoformat(a.end)
    pipe = TennisDailyPipeline()
    weeks = list(mondays(start, end))
    print(f"Backfill {len(weeks)} settimane {start} -> {end}")
    for i, monday in enumerate(weeks, 1):
        recs = pipe._update_recent_results(monday)
        print(f"  [{i}/{len(weeks)}] {monday}: store {len(recs)} risultati")
        time.sleep(1)
    print("Store:", pipe._recent_store_path(), f"({len(pipe._load_recent_store())} record)")


if __name__ == "__main__":
    main()
