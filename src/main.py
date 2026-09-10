#!/usr/bin/env python3
"""
TennisPreview - Tennis Match Analysis and Value Betting System

Scans upcoming matches, collects data from multiple sources (historical
databases, odds APIs, news feeds), runs prediction models, and shows
probability estimates plus potential value bets for every event.
"""

import asyncio
import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from src.utils.config import config
from src.utils.logging import setup_logging, get_logger, progress
from src.pipeline.daily import TennisDailyPipeline, PipelineResult

logger = get_logger("main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TennisPreview - Tennis Match Analysis & Value Betting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main                    # Run for today (live, real odds)
  python -m src.main --date 2024-01-15  # Run for specific date
  python -m src.main --days 7           # Run for next 7 days
  python -m src.main --update-data      # Only update historical data
  python -m src.main --demo             # Offline sandbox (simulated odds)

The program outputs decisions only - it never bets automatically
and handles no paper money.
        """
    )
    
    parser.add_argument(
        '--date', '-d',
        type=str,
        help='Target date (YYYY-MM-DD), default: today'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=1,
        help='Number of days to run (default: 1)'
    )
    
    parser.add_argument(
        '--update-data',
        action='store_true',
        help='Only update historical data, no analysis'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Offline demo mode using historical data only'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Log level (default: INFO)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars'
    )

    return parser.parse_args()


def _wrap(text: str, width: int = 90, prefix: str = "      ") -> list:
    """Wrap a long reason string for console display."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + 1 + len(w) > width:
            lines.append(prefix + cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(prefix + cur)
    return lines


def print_results(result: PipelineResult, target_date: date):
    """Print full analysis results to console"""
    line = "=" * 96
    
    print()
    print(line)
    if result.demo_mode:
        print(f"  RESULTS FOR {target_date}  (DEMO MODE - dati simulati)")
    else:
        n_odds = len(result.odds)
        n_m = len(result.matches)
        tag = f" (LIVE - quote reali {n_odds}/{n_m})" if n_m else ""
        print(f"  RESULTS FOR {target_date}{tag}")
        if n_m and not n_odds:
            print("  NESSUNA QUOTA REALE DISPONIBILE - solo pronostici, nessun bet")
    print(line)
    
    if not result.matches:
        print("\n  No matches found for this date.")
        return
    
    # Header
    print()
    print(f"  {'Match':<34} {'Tier':<4} {'P1 Model':>8} {'P2 Model':>8} "
          f"{'P1 Mkt':>7} {'P2 Mkt':>7} {'O1':>6} {'O2':>6}  Signal")
    print("  " + "-" * 92)
    
    # Rows for every analyzed event
    for m in result.matches:
        pred = result.predictions.get(m.id)
        odds = result.odds.get(m.id)
        
        if not pred:
            continue
        
        tier = pred.tier.value if hasattr(pred.tier, 'value') else pred.tier
        
        mkt1 = f"{odds.implied_prob_home*100:5.1f}%" if odds else "   n/a"
        mkt2 = f"{odds.implied_prob_away*100:5.1f}%" if odds else "   n/a"
        o1 = f"{odds.best_odds_home:.2f}" if odds else "  n/a"
        o2 = f"{odds.best_odds_away:.2f}" if odds else "  n/a"
        
        # Determine signal
        signal = "      -"
        for d in result.decisions:
            if d.value_bet.match_id == m.id:
                side = "P1" if d.value_bet.side == "player1" else "P2"
                signal = f"  DEC {side} ({d.value_bet.edge*100:+.1f}%)"
                break
        
        print(f"  {m.match_key[:33]:<34} {tier:>4} "
              f"{pred.p1_win_prob*100:7.1f}% {pred.p2_win_prob*100:7.1f}% "
              f"{mkt1:>7} {mkt2:>7} {o1:>6} {o2:>6} {signal}")
    
    print()
    
    # Decisions detail (no money: decisions only)
    if result.decisions:
        print(f"  {'DECISIONI FINALI':-^92}")
        print()
        print(f"  {'#':<3} {'Match':<32} {'Scelta':<20} {'Quota':>6} "
              f"{'Book':<12} {'Edge':>7} {'Tier':>4} {'Sisal':>5}")
        print("  " + "-" * 92)

        for d in result.decisions:
            vb = d.value_bet
            tier = vb.tier.value if hasattr(vb.tier, 'value') else vb.tier
            print(f"  {d.rank:<3} {vb.match_key[:31]:<32} {vb.player_name[:19]:<20} "
                  f"{vb.odds:>6.2f} {d.bookmaker[:11]:<12} "
                  f"{vb.edge*100:>+6.1f}% {tier:>4} {d.sisal:>5}")

        print()
        print("  PERCHE' (dati veritieri: storico + news + mercato):")
        for d in result.decisions:
            vb = d.value_bet
            print(f"  [{d.rank}] {vb.player_name} ({vb.match_key})")
            for line in _wrap(d.reason):
                print(line)
        print()
    else:
        print("  Nessuna decisione (nessun edge sopra soglia).")
        print()

    # Excluded value (transparency)
    if result.excluded:
        print(f"  {'VALUE SCARTATE':-^92}")
        for d in result.excluded:
            vb = d.value_bet
            print(f"  - {vb.player_name} ({vb.match_key}) "
                  f"edge {vb.edge*100:+.1f}% -> {d.excluded_reason}")
        print()

    # Decision summary (counts only, no money)
    print(f"  {'RIEPILOGO':-^92}")
    summ = getattr(result, "decision_summary", {}) or {}
    print(f"  Match analizzati:      {len(result.matches):>10d}")
    print(f"  Value trovate:         {summ.get('value_found', len(result.value_bets)):>10d}")
    print(f"  Decisioni finali:      {summ.get('decisions', len(result.decisions)):>10d}")
    print(f"  Scartate:              {summ.get('excluded', len(result.excluded)):>10d}")
    print(f"  Su Sisal:              {summ.get('sisal_yes', 0):>10d}")
    print(f"  Per tier:              {summ.get('by_tier', {})}")
    print()


async def run_single_day(pipeline: TennisDailyPipeline, target_date: date) -> int:
    """Run pipeline for a single day"""
    logger.info(f"{'='*60}")
    logger.info(f"Processing {target_date}")
    logger.info(f"{'='*60}")
    
    try:
        result = await pipeline.run(target_date)
        print_results(result, target_date)
        return len(result.decisions)
    except Exception as e:
        logger.error(f"Failed to process {target_date}: {e}", exc_info=True)
        return 0


async def main():
    args = parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Load config
    if args.config:
        config.load(args.config)
    
    # Disable progress bars if requested
    if args.no_progress:
        progress.set_enabled(False)

    print()
    print("=" * 96)
    print("  TennisPreview - Tennis Match Analysis & Decisioni Value")
    print("  (solo decisioni, niente soldi: non scommette automaticamente)")
    print("=" * 96)
    logger.info("TennisPreview starting.")

    if args.update_data:
        logger.info("Running data update only...")
        pipeline = TennisDailyPipeline()
        await pipeline._load_historical_data()
        logger.info("Data update complete")
        return

    # Parse target date
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    else:
        target_date = date.today()

    # Initialize pipeline
    pipeline = TennisDailyPipeline(demo_mode=args.demo)

    total_decisions = 0

    if args.days == 1:
        # Single day
        total_decisions = await run_single_day(pipeline, target_date)
    else:
        # Multiple days
        for i in range(args.days):
            current_date = target_date + timedelta(days=i)
            decs = await run_single_day(pipeline, current_date)
            total_decisions += decs

            # Small delay between days
            if i < args.days - 1:
                await asyncio.sleep(2)

    # Final summary (counts only, no money)
    print()
    print("=" * 96)
    print("  FINAL SUMMARY")
    print("=" * 96)
    print(f"  Total days processed : {args.days}")
    print(f"  Total decisioni      : {total_decisions}")
    print(f"  Mode                 : {'DEMO (offline)' if args.demo else 'LIVE'}")

    print()
    logger.info("TennisPreview finished")


if __name__ == '__main__':
    asyncio.run(main())