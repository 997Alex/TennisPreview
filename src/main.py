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
import tkinter as tk
from datetime import date, datetime, timedelta
from pathlib import Path

from src.utils.config import config
from src.utils.logging import setup_logging, get_logger, progress
from src.pipeline.daily import TennisDailyPipeline, PipelineResult
from src.gui.match_analyzer import MatchAnalyzerApp

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
    
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Launch GUI interface for single match analysis'
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
    
    # Tabella semplice: solo favorito + % vittoria (il resto resta nel motore).
    # Tier/quote/mercato/edge restano usati dall'algoritmo e nel report JSON.
    sisal_all = getattr(result, "sisal_flag", {}) or {}
    print()
    print(f"  {'Match':<38} {'Favorito':<26} {'Prob.':>6}  {'Sisal':<3}")
    print("  " + "-" * 80)

    for m in result.matches:
        pred = result.predictions.get(m.id)
        if not pred:
            continue
        if pred.p1_win_prob >= pred.p2_win_prob:
            fav, fav_pct = (m.player1.name if m.player1 else "?"), pred.p1_win_prob * 100
        else:
            fav, fav_pct = (m.player2.name if m.player2 else "?"), pred.p2_win_prob * 100
        sflag = sisal_all.get(m.id, ("?", ""))[0]
        mark = "*" if any(d.value_bet.match_id == m.id for d in result.decisions) else " "
        print(f" {mark} {m.match_key[:37]:<38} {fav[:25]:<26} {fav_pct:5.1f}%  {sflag:<3}")

    print()
    print("  (*) = tra le decisioni finali (vedi sotto con motivazione)")

    # Per-match analysis: % successo + segnali extra-match per OGNI partita
    print(f"  {'ANALISI PER MATCH (% successo + segnali)':-^92}")
    print()
    news_all = getattr(result, "news", {}) or {}
    for m in result.matches:
        pred = result.predictions.get(m.id)
        odds = result.odds.get(m.id)
        feat = result.features.get(m.id)
        if not pred:
            continue
        p1, p2 = pred.p1_win_prob * 100, pred.p2_win_prob * 100
        fav = m.player1.name if p1 >= p2 else m.player2.name
        fav_pct = max(p1, p2)
        if odds:
            q1, q2 = odds.best_odds_home, odds.best_odds_away
            book = odds.best_book_home if p1 >= p2 else odds.best_book_away
            qfav = q1 if p1 >= p2 else q2
            odds_txt = f"quota {fav} @{qfav:.2f} ({book})"
        else:
            odds_txt = "quota n/a (non su Sisal/book)"
        h2h_txt = "-"
        if feat and feat.h2h_total > 0:
            h2h_txt = (f"{feat.h2h_p1_wins}-{feat.h2h_total - feat.h2h_p1_wins} "
                       f"({feat.h2h_p1_wins_surface}-{feat.h2h_surface - feat.h2h_p1_wins_surface} superficie)")
        n1 = (news_all.get(m.player1.name, {}) if m.player1 else {})
        n2 = (news_all.get(m.player2.name, {}) if m.player2 else {})
        def _ntxt(n):
            if not n or not n.get("article_count"):
                return "neutre"
            sign = "+" if n.get("impact", 0) > 0.05 else ("-" if n.get("impact", 0) < -0.05 else "~")
            return f"{sign} impatto {n.get('impact', 0):+.2f} ({n.get('article_count')} art.)"
        poly_txt = "-"
        poly = (result.poly or {}).get(m.id)
        if poly:
            poly_txt = f"crowd {poly[0]:.1%} vs {poly[1]:.1%}"
        sflag, snote = sisal_all.get(m.id, ("?", ""))
        sisal_txt = f"sisal: {sflag}" + (f" ({snote})" if snote and sflag == "no" else "")
        print(f"  {m.match_key[:60]}  [{sisal_txt}]")
        print(f"    pronostico: {m.player1.name[:22]:<22} {p1:5.1f}%  |  "
              f"{m.player2.name[:22]:<22} {p2:5.1f}%  -> favorito {fav} ({fav_pct:.1f}%)")
        print(f"    sisal/mercato: {odds_txt}  |  H2H: {h2h_txt}")
        print(f"    news {m.player1.name[:18]:<18}: {_ntxt(n1)}  |  "
              f"{m.player2.name[:18]:<18}: {_ntxt(n2)}  |  polymarket: {poly_txt}")
    print()

    # Decisions detail (no money: decisions only)
    if result.decisions:
        print(f"  {'DECISIONI FINALI':-^92}")
        print()
        print(f"  {'#':<3} {'Match':<28} {'PUNTARE SU':<24} {'Quota':>6} "
              f"{'Book':<10} {'Edge':>7} {'Tier':>4} {'Sisal':>5}")
        print("  " + "-" * 92)

        for d in result.decisions:
            vb = d.value_bet
            tier = vb.tier.value if hasattr(vb.tier, 'value') else vb.tier
            print(f"  {d.rank:<3} {vb.match_key[:27]:<28} {vb.player_name[:23]:<24} "
                  f"{vb.odds:>6.2f} {d.bookmaker[:9]:<10} "
                  f"{vb.edge*100:>+6.1f}% {tier:>4} {d.sisal:>5}")

        print()
        print("  PERCHE' (dati veritieri: storico + news + mercato):")
        for d in result.decisions:
            vb = d.value_bet
            print(f"  [{d.rank}] PUNTARE SU {vb.player_name} @ {vb.odds:.2f} "
                  f"({d.bookmaker}) - {vb.match_key}")
            for line in _wrap(d.reason):
                print(line)
        print()
    else:
        print("  Nessuna decisione (nessun edge sopra soglia).")
        print()

    # Favoriti confermati: prob modello alta (>=70%) con quota reale,
    # anche senza edge value. Informativi: il mercato li prezza il giusto.
    favs = []
    for m in result.matches:
        pred = result.predictions.get(m.id)
        odds = result.odds.get(m.id)
        if not pred or not odds:
            continue
        if pred.p1_win_prob >= pred.p2_win_prob:
            name, prob, q, book = (m.player1.name if m.player1 else "?",
                                   pred.p1_win_prob, odds.best_odds_home,
                                   odds.best_book_home)
        else:
            name, prob, q, book = (m.player2.name if m.player2 else "?",
                                   pred.p2_win_prob, odds.best_odds_away,
                                   odds.best_book_away)
        if prob >= 0.70 and q > 1.0:
            favs.append((prob, m.match_key, name, q, book))
    if favs:
        print(f"  {'FAVORITI CONFERMATI (prob alta, non value)':-^92}")
        for prob, key, name, q, book in sorted(favs, reverse=True):
            print(f"  - {name[:26]:<26} {prob*100:5.1f}%  @{q:.2f} ({book})  [{key[:40]}]")
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

    # Windows: forza stdout UTF-8 (console cp1252 crasherebbe su ►/accenti)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # GUI mode
    if args.gui:
        root = tk.Tk()
        app = MatchAnalyzerApp(root)
        root.mainloop()
        return

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