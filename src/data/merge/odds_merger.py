"""Odds merger: unisce snapshot multi-book in MergedOdds."""
from datetime import datetime

from src.utils.logging import get_logger
from src.utils.models import MergedOdds

logger = get_logger("odds_merger")

__all__ = ["MergedOdds", "OddsMerger"]


class OddsMerger:
    def merge(self, all_odds: dict[str, list]) -> dict[str, MergedOdds]:
        out: dict[str, MergedOdds] = {}
        for mid, snaps in (all_odds or {}).items():
            try:
                m = self._merge_match(mid, snaps)
                if m:
                    out[mid] = m
            except Exception as e:
                logger.warning(f"merge {mid} failed: {e}")
        return out

    def _merge_match(self, match_id: str, snapshots: list) -> MergedOdds:
        snaps = [s for s in (snapshots or [])
                 if s and getattr(s, "odds_home", 0) > 1.0
                 and getattr(s, "odds_away", 0) > 1.0]
        if not snaps:
            return None  # type: ignore[return-value]
        best_h = max(snaps, key=lambda s: s.odds_home)
        best_a = max(snaps, key=lambda s: s.odds_away)
        avg_h = sum(s.odds_home for s in snaps) / len(snaps)
        avg_a = sum(s.odds_away for s in snaps) / len(snaps)
        ip_h, ip_a = 1 / best_h.odds_home, 1 / best_a.odds_away
        tot = ip_h + ip_a
        vig = tot - 1.0
        if tot > 0:
            ip_h, ip_a = ip_h / tot, ip_a / tot
        return MergedOdds(
            match_id=match_id,
            best_odds_home=round(best_h.odds_home, 2),
            best_odds_away=round(best_a.odds_away, 2),
            best_book_home=str(getattr(best_h, "bookmaker", "?")),
            best_book_away=str(getattr(best_a, "bookmaker", "?")),
            implied_prob_home=ip_h, implied_prob_away=ip_a,
            vig=vig, avg_odds_home=round(avg_h, 2),
            avg_odds_away=round(avg_a, 2),
            sources_count=len(snaps),
            last_update=datetime.now(),
        )
