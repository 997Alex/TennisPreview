"""ESPN tennis scoreboard client (gratuito, senza chiave).

API pubblica: https://site.api.espn.com/apis/site/v2/sports/tennis/<league>/scoreboard
league: atp / wta. Parametro dates=YYYYMMDD.
"""
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import requests

from src.utils.logging import get_logger

logger = get_logger("espn")


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)
    )


def normalize_name(name: str) -> str:
    """Chiave stabile 'cognome nome' minuscola senza accenti/punteggiatura."""
    toks = name_tokens(name)
    return " ".join(toks)


def name_tokens(name: str) -> list[str]:
    s = _strip_accents(str(name or "").lower())
    s = re.sub(r"[^a-z ]", " ", s)
    return [t for t in s.split() if t]


def surname_key(name: str) -> str:
    toks = name_tokens(name)
    if not toks:
        return ""
    return "".join(toks[1:]) if len(toks) > 1 else "".join(toks)


@dataclass
class ESPNMatch:
    id: str
    tournament: str
    league: str = "atp"
    round: str = "TBD"
    player1: str = ""
    player2: str = ""
    status: str = "pre"  # pre | in | post
    surface_hint: str = "Hard"
    level_hint: str = ""
    best_of: int = 3
    scheduled_time: datetime | None = None


@dataclass
class ESPNResult:
    uid: str
    tournament: str
    league: str = "atp"
    round: str = ""
    best_of: int = 3
    surface: str = "Hard"
    level: str = ""
    winner: str = ""
    loser: str = ""
    date: Optional[date] = None  # noqa: UP045 - 'date | None' non valido: il campo si chiama come il tipo


_LEVEL_MAP = {
    "grand slam": "GS", "masters 1000": "MS1000", "wta 1000": "WTA1000",
    "atp 500": "ATP500", "wta 500": "WTA500", "atp 250": "ATP250",
    "wta 250": "WTA250", "wta 125": "WTA125", "challenger": "CH100",
}


def _guess_level(tournament: str) -> str:
    t = (tournament or "").lower()
    for k, v in _LEVEL_MAP.items():
        if k in t:
            return v
    return ""


class ESPNClient:
    BASE = "https://site.api.espn.com/apis/site/v2/sports/tennis"

    def _fetch(self, league: str, target: date) -> list[dict]:
        try:
            r = requests.get(
                f"{self.BASE}/{league}/scoreboard",
                params={"dates": target.strftime("%Y%m%d")},
                timeout=25,
            )
            r.raise_for_status()
            return r.json().get("events", [])
        except Exception as e:
            logger.warning(f"ESPN {league} {target} failed: {e}")
            return []

    @staticmethod
    def _iter_competitions(ev: dict):
        """I match sono annidati in event -> groupings[] -> competitions[]."""
        for grouping in ev.get("groupings", []) or []:
            yield from grouping.get("competitions", []) or []

    @classmethod
    def _parse_competition(cls, ev: dict, comp: dict, league: str,
                           tourn: str) -> ESPNMatch | None:
        try:
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                return None
            names = []
            for c in competitors[:2]:
                a = (c.get("athlete") or {}).get("displayName", "")
                names.append(str(a).strip())
            if not names[0] or not names[1]:
                return None
            if names[0].upper() in ("TBD", "UNKNOWN", "TBA") or names[1].upper() in ("TBD", "UNKNOWN", "TBA"):
                return None
            status = (comp.get("status") or {}).get("type", {}).get("state", "pre")
            st = {"pre": "pre", "in": "in", "post": "post"}.get(status, "pre")
            try:
                dt = datetime.fromisoformat(
                    str(comp.get("date") or ev.get("date")).replace("Z", "+00:00"))
                dt = dt.replace(tzinfo=None)
            except Exception:
                dt = datetime.now()
            notes = comp.get("notes", "")
            if isinstance(notes, list):
                notes = str(notes[0] if notes else "TBD")
            periods = ((comp.get("format") or {}).get("regulation") or {}).get("periods", 3)
            return ESPNMatch(
                id=f"espn_{comp.get('id')}",
                tournament=tourn,
                league=league,
                round=str(notes)[:20] or "TBD",
                player1=names[0], player2=names[1],
                status=st,
                surface_hint="Hard",
                level_hint=_guess_level(tourn),
                best_of=5 if (periods == 5 or ("grand slam" in tourn.lower()
                                              and league == "atp")) else 3,
                scheduled_time=dt,
            )
        except Exception:
            return None

    def get_matches_for_date(self, target: date) -> list[ESPNMatch]:
        out, seen = [], set()
        for league in ("atp", "wta"):
            for ev in self._fetch(league, target):
                tourn = str(ev.get("name", "ESPN event"))
                for comp in self._iter_competitions(ev):
                    m = self._parse_competition(ev, comp, league, tourn)
                    if m and m.status in ("pre", "in") and m.id not in seen:
                        seen.add(m.id)
                        out.append(m)
        logger.info(f"ESPN: {len(out)} match per {target}")
        return out

    def get_results_for_date(self, target: date) -> list[ESPNResult]:
        out = []
        for league in ("atp", "wta"):
            for ev in self._fetch(league, target):
                tourn = str(ev.get("name", ""))
                for comp in self._iter_competitions(ev):
                    try:
                        competitors = comp.get("competitors", [])
                        state = (comp.get("status") or {}).get("type", {}).get("state", "")
                        if state != "post" or len(competitors) < 2:
                            continue
                        win = [c for c in competitors if c.get("winner")]
                        if len(win) != 1:
                            continue
                        loser = [c for c in competitors if not c.get("winner")][0]
                        winner = str((win[0].get("athlete") or {}).get("displayName", ""))
                        lname = str((loser.get("athlete") or {}).get("displayName", ""))
                        if not winner or not lname:
                            continue
                        try:
                            cdate = datetime.fromisoformat(
                                str(comp.get("date")).replace("Z", "+00:00")).date()
                        except Exception:
                            continue
                        if cdate != target:
                            continue
                        out.append(ESPNResult(
                            uid=f"espn_{comp.get('id')}",
                            tournament=tourn, league=league,
                            winner=winner, loser=lname,
                            surface="Hard", level=_guess_level(tourn),
                            date=cdate,
                        ))
                    except Exception:
                        continue
        return out
