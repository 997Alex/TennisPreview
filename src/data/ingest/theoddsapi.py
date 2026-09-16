"""TheOddsAPI client (quote reali con chiave gratuita: 500 req/mese).

Chiave su https://the-odds-api.com/ -> .env THEODDSAPI_KEY.
Senza chiave tutti i metodi ritornano [] senza bloccare la pipeline.
"""
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime

import requests

from src.utils.logging import get_logger

logger = get_logger("theoddsapi")


@dataclass
class OddsApiMatch:
    id: str
    home_team: str
    away_team: str
    commence_time: datetime


def _norm(s: str) -> str:
    s = "".join(
        c for c in unicodedata.normalize("NFKD", (s or "").lower())
        if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z ]", " ", s)


class TheOddsAPIClient:
    BASE = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("THEODDSAPI_KEY", "")

    def _valid(self) -> bool:
        k = (self.api_key or "").strip()
        return bool(k) and k not in ("your_key_here",) and len(k) > 8

    def _tennis_keys(self) -> list[str]:
        try:
            r = requests.get(f"{self.BASE}/sports",
                             params={"apiKey": self.api_key}, timeout=25)
            r.raise_for_status()
            return [s.get("key", "") for s in r.json()
                    if "tennis" in str(s.get("key", "")).lower()]
        except Exception as e:
            logger.warning(f"TheOddsAPI sports failed: {e}")
            return []

    def get_tennis_odds(self) -> list[OddsApiMatch]:
        """Fixture per discovery match (nomi + orari)."""
        out: list[OddsApiMatch] = []
        for ev in self._iter_events():
            try:
                out.append(OddsApiMatch(
                    id=str(ev["id"]), home_team=ev["home"], away_team=ev["away"],
                    commence_time=ev["time"]))
            except Exception:
                continue
        return out

    def get_odds_by_names(self) -> dict[tuple[str, str], list]:
        """Quote reali indicizzate per coppia nomi normalizzata (entrambi i versi)."""
        from datetime import datetime as _dt

        from src.utils.models import OddsSnapshot as _Snap
        out: dict[tuple[str, str], list] = {}
        for ev in self._iter_events():
            try:
                a, b = _norm(ev["home"]), _norm(ev["away"])
                if not a or not b:
                    continue
                for bm, o1, o2 in ev["prices"]:
                    snap = _Snap(bookmaker=f"theoddsapi:{bm}", market="h2h",
                                 odds_home=float(o1), odds_away=float(o2),
                                 timestamp=_dt.now(), source="theoddsapi")
                    out.setdefault((a, b), []).append(snap)
            except Exception:
                continue
        logger.info(f"TheOddsAPI: quote per {len(out) // 2} eventi")
        return out

    def _iter_events(self) -> list[dict]:
        """Eventi tennis con prezzi h2h (una chiamata API per sport)."""
        if not self._valid():
            return []
        events: list[dict] = []
        try:
            keys = self._tennis_keys()
        except Exception:
            return []
        for key in keys[:12]:
            try:
                r = requests.get(
                    f"{self.BASE}/sports/{key}/odds",
                    params={"apiKey": self.api_key, "regions": "eu",
                            "markets": "h2h", "oddsFormat": "decimal"},
                    timeout=30)
                if r.status_code == 422:
                    continue  # sport senza mercato h2h
                r.raise_for_status()
                for ev in r.json():
                    try:
                        home = str(ev.get("home_team", ""))
                        away = str(ev.get("away_team", ""))
                        prices = []
                        for bm in ev.get("bookmakers", []) or []:
                            for mk in bm.get("markets", []) or []:
                                if mk.get("key") != "h2h":
                                    continue
                                outs = {o.get("name"): o.get("price")
                                        for o in mk.get("outcomes", []) or []}
                                if home in outs and away in outs:
                                    prices.append((str(bm.get("title", "book")),
                                                   float(outs[home]), float(outs[away])))
                        if not prices:
                            continue
                        try:
                            ct = datetime.fromisoformat(
                                str(ev.get("commence_time")).replace("Z", "+00:00"))
                            ct = ct.replace(tzinfo=None)
                        except Exception:
                            ct = datetime.now()
                        events.append({"id": str(ev.get("id")), "home": home,
                                       "away": away, "time": ct, "prices": prices})
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"TheOddsAPI {key} failed: {e}")
        return events

    def get_odds_for_matches(self, match_ids) -> dict[str, list]:
        # mapping per nomi gestito dalla pipeline (get_odds_by_names)
        return {}
