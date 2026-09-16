"""Sackmann historical DB ingester (offline-safe stub).

Interfaccia richiesta da src/pipeline/daily.py:
  update_repos(), load_matches(tour, years), load_rankings(tour),
  build_player_profiles(matches_df, rankings_df),
  parse_surface(str), parse_tournament_level(code, tour)

Se la cartella data/sackmann/ contiene i CSV di JeffSackmann li usa,
altrimenti restituisce DataFrame vuoti e la pipeline prosegue con
profili parziali (default Elo 1500) invece di crashare.
"""
from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger
from src.utils.models import Player, Surface, TournamentLevel

logger = get_logger("sackmann")


class SackmannIngester:
    def __init__(self, local_path: str = "data/sackmann"):
        self.local_path = Path(local_path)

    # -- repos ---------------------------------------------------------
    def update_repos(self) -> None:
        """Best-effort: senza git non fa nulla (log + return)."""
        logger.info("Sackmann update_repos: skip (usa data/sackmann se presente)")

    # -- loading --------------------------------------------------------
    def load_matches(self, tour: str = "atp", years=None) -> pd.DataFrame | None:
        frames = []
        years = years or []
        for y in years:
            for cand in [
                self.local_path / f"{tour}_matches_{y}.csv",
                self.local_path / tour / f"{tour}_matches_{y}.csv",
            ]:
                if cand.exists():
                    try:
                        frames.append(pd.read_csv(cand, low_memory=False))
                    except Exception as e:
                        logger.warning(f"Sackmann read failed {cand}: {e}")
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df["tour"] = tour.upper()
            return df
        return pd.DataFrame()

    def load_rankings(self, tour: str = "atp") -> pd.DataFrame | None:
        for cand in [
            self.local_path / f"{tour}_rankings_current.csv",
            self.local_path / tour / f"{tour}_rankings_current.csv",
        ]:
            if cand.exists():
                try:
                    return pd.read_csv(cand, low_memory=False)
                except Exception as e:
                    logger.warning(f"Sackmann rankings read failed {cand}: {e}")
        return pd.DataFrame()

    # -- profiles ---------------------------------------------------------
    def build_player_profiles(self, matches_df, rankings_df=None) -> dict[str, Player]:
        """Profili minimi: win% superficie + Elo superficie (k=32) + forma ultime 10."""
        db: dict[str, Player] = {}
        try:
            if matches_df is None or getattr(matches_df, "empty", True):
                return db
            recs = matches_df.to_dict("records")
            for m in recs:
                try:
                    wid = str(m.get("winner_id", "")).strip()
                    lid = str(m.get("loser_id", "")).strip()
                    if not wid or not lid or wid == "nan" or lid == "nan":
                        continue
                    wname = str(m.get("winner_name", wid))
                    lname = str(m.get("loser_name", lid))
                    surface = self.parse_surface(m.get("surface", "Hard"))
                    for pid, pname in ((wid, wname), (lid, lname)):
                        if pid not in db:
                            db[pid] = Player(id=pid, name=pname)
                    wpl, lpl = db[wid], db[lid]
                    for pl, won in ((wpl, True), (lpl, False)):
                        if surface not in pl.surface_matches:
                            pl.surface_matches[surface] = 0
                            pl.surface_win_pct[surface] = 0.0
                            pl.surface_elo[surface] = 1500.0
                        pl.surface_matches[surface] += 1
                        prev = pl.surface_win_pct[surface] * (pl.surface_matches[surface] - 1)
                        if won:
                            prev += 1
                        pl.surface_win_pct[surface] = prev / pl.surface_matches[surface]
                        buf = getattr(pl, "_recent_results", None)
                        if buf is None:
                            buf = [1] * pl.wins_last_10 + [0] * pl.losses_last_10
                        buf.append(1 if won else 0)
                        if len(buf) > 10:
                            buf = buf[-10:]
                        pl._recent_results = buf  # type: ignore[attr-defined]
                        pl.wins_last_10 = sum(buf)
                        pl.losses_last_10 = len(buf) - pl.wins_last_10
                    we = wpl.surface_elo.get(surface, 1500.0)
                    le = lpl.surface_elo.get(surface, 1500.0)
                    ew = 1.0 / (1.0 + 10 ** ((le - we) / 400.0))
                    wpl.surface_elo[surface] = we + 32.0 * (1 - ew)
                    lpl.surface_elo[surface] = le + 32.0 * (0 - (1 - ew))
                except Exception:
                    continue
            for pl in db.values():
                if hasattr(pl, "_recent_results"):
                    try:
                        delattr(pl, "_recent_results")
                    except AttributeError:
                        pass
        except Exception as e:
            logger.warning(f"build_player_profiles failed: {e}")
        logger.info(f"Sackmann profiles built: {len(db)}")
        return db

    # -- parsers ------------------------------------------------------------
    @staticmethod
    def parse_surface(value) -> Surface:
        try:
            if isinstance(value, Surface):
                return value
            s = str(value or "Hard").strip().lower()
            if "clay" in s:
                return Surface.CLAY
            if "grass" in s:
                return Surface.GRASS
            if "carpet" in s:
                return Surface.CARPET
            if "indoor" in s or s == "i":
                return Surface.HARD_INDOOR
            return Surface.HARD
        except Exception:
            return Surface.HARD

    @staticmethod
    def parse_tournament_level(code, tour: str = "atp") -> TournamentLevel:
        try:
            c = str(code or "").strip().upper()
            t = str(tour or "atp").lower()
            if c == "G":
                return TournamentLevel.GS
            if c == "M":
                return TournamentLevel.MS1000 if t == "atp" else TournamentLevel.WTA1000
            if c == "A":
                return TournamentLevel.ATP500 if t == "atp" else TournamentLevel.WTA500
            if c == "B":
                return TournamentLevel.ATP250 if t == "atp" else TournamentLevel.WTA250
            if c == "C":
                return TournamentLevel.CH100
            if c == "F":
                return TournamentLevel.CH75
            mapping = {
                "GS": TournamentLevel.GS, "MS1000": TournamentLevel.MS1000,
                "ATP500": TournamentLevel.ATP500, "ATP250": TournamentLevel.ATP250,
                "WTA1000": TournamentLevel.WTA1000, "WTA500": TournamentLevel.WTA500,
                "WTA250": TournamentLevel.WTA250, "WTA125": TournamentLevel.WTA125,
            }
            return mapping.get(c, TournamentLevel.UNKNOWN)
        except Exception:
            return TournamentLevel.UNKNOWN
