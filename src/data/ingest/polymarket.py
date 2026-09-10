"""Polymarket sentiment client (Gamma public search API, gratis senza chiave)."""
import re
import unicodedata

import requests

from src.utils.logging import get_logger

logger = get_logger("polymarket")

GAMMA_SEARCH = "https://gamma-api.polymarket.com/public-search"
CLOB_PRICE = "https://clob.polymarket.com/price"


def _norm(s: str) -> str:
    s = "".join(
        c for c in unicodedata.normalize("NFKD", (s or "").lower())
        if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z ]", " ", s)


class PolymarketClient:
    def __init__(self, timeout: int = 20):
        self.timeout = timeout

    def _find_h2h(self, p1: str, p2: str) -> tuple[float, float] | None:
        """Cerca mercato con entrambi i cognomi, ritorna (p1_prob, p2_prob)."""
        n1, n2 = _norm(p1).split(), _norm(p2).split()
        surn1 = n1[-1] if n1 else ""
        surn2 = n2[-1] if n2 else ""
        if len(surn1) < 3 or len(surn2) < 3:
            return None
        for query in (f"{surn1} {surn2}", f"{surn1} vs {surn2}"):
            try:
                r = requests.get(GAMMA_SEARCH, params={"q": query},
                                 timeout=self.timeout)
                if not r.ok:
                    continue
                data = r.json()
                events = data if isinstance(data, list) else data.get("events", [])
                for ev in events[:8]:
                    try:
                        txt = _norm(f"{ev.get('title', '')} {ev.get('slug', '')}")
                        if surn1 not in txt or surn2 not in txt:
                            continue
                        prob = self._event_prob(ev, surn1, surn2)
                        if prob:
                            return prob
                    except Exception:
                        continue
            except Exception as ex:
                logger.warning(f"Polymarket search failed: {ex}")
        return None

    def _event_prob(self, ev: dict, surn1: str, surn2: str
                    ) -> tuple[float, float] | None:
        markets = ev.get("markets", []) or []
        for m in markets:
            try:
                outcomes = m.get("outcomes")
                prices = m.get("outcomePrices")
                if isinstance(outcomes, str):
                    import json as _j
                    outcomes = _j.loads(outcomes)
                    prices = _j.loads(prices)
                if not outcomes or not prices or len(outcomes) < 2:
                    continue
                o0 = _norm(str(outcomes[0]))
                p0 = float(prices[0])
                if not (0.02 < p0 < 0.98):
                    continue
                # outcomes[0] si riferisce a p1 o p2?
                if surn1 in o0 and surn2 not in o0:
                    return (p0, 1 - p0)
                if surn2 in o0 and surn1 not in o0:
                    return (1 - p0, p0)
            except Exception:
                continue
        return None

    def get_odds_for_matches(self, matches: list) -> dict[str, list]:
        """Ritorna {match_id: [OddsSnapshot-like]} per i match coperti."""
        from datetime import datetime as _dt

        from src.utils.models import OddsSnapshot as _Snap
        out: dict[str, list] = {}
        for m in matches:
            try:
                if not m.player1 or not m.player2:
                    continue
                found = self._find_h2h(m.player1.name, m.player2.name)
                if not found:
                    continue
                p1, p2 = found
                if p1 <= 0 or p2 <= 0:
                    continue
                out[m.id] = [_Snap(
                    bookmaker="polymarket", market="h2h",
                    odds_home=round(1 / p1, 2), odds_away=round(1 / p2, 2),
                    timestamp=_dt.now(), source="polymarket",
                )]
            except Exception:
                continue
        logger.info(f"Polymarket: {len(out)} match coperti")
        return out
