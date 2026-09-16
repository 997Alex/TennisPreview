"""News ingester: RSS tennis + Google News per giocatore.

Per ogni giocatore calcola impatto in [-1, +1]:
  negativo: injury/infortunio, illness/malattia, withdraw/ritiro,
            surgery/operazione, sick, doubt/in dubbio
  positivo: title/vince, comeback, fit/recuperato, final/finale
"""
import re
import unicodedata
from urllib.parse import quote_plus

import feedparser
import requests

from src.utils.logging import get_logger

logger = get_logger("news")

TENNIS_FEEDS = [
    "https://www.atptour.com/en/media/rss/news.xml",
    "https://www.wtatennis.com/news/rss",
    "https://www.tennis.com/pro-game/rss/",
]

NEG_PATTERNS = [
    (r"injur\w*|infortun\w*", -0.6),
    (r"withdraw\w*|ritir\w*|pulls?\s*out", -0.5),
    (r"illness|malattia|sick|virus|fever|febbre", -0.4),
    (r"surgery|operazione|operato", -0.6),
    (r"doubt|doubtful|in dubbio|uncertain", -0.3),
    (r"personal|family|famiglia", -0.2),
    (r"ban|suspend|squalific", -0.5),
]

POS_PATTERNS = [
    (r"\bwin\b|\bvince\b|title|titolo|champion|campione", 0.3),
    (r"comeback|rimonta|return.*tour|torna", 0.25),
    (r"\bfit\b|recuper\w*|recovered|100%", 0.3),
    (r"final\b|finale|semifinal|semifinale", 0.15),
]


def _norm(s: str) -> str:
    s = "".join(
        c for c in unicodedata.normalize("NFKD", (s or "").lower())
        if not unicodedata.combining(c)
    )
    return re.sub(r"\s+", " ", s).strip()


class NewsIngester:
    def __init__(self, feeds: list[str] = None):
        self.feeds = feeds or list(TENNIS_FEEDS)
        self._player_cache: dict = {}  # query Google News per giocatore (per run)

    def fetch_all_feeds(self, hours_back: int = 72) -> list[dict]:
        articles: list[dict] = []
        for url in self.feeds:
            try:
                fp = feedparser.parse(url)
                for e in fp.entries[:60]:
                    try:
                        txt = f"{e.get('title', '')} {e.get('summary', '')}"
                        articles.append({
                            "title": e.get("title", ""),
                            "text": txt,
                            "published": str(e.get("published", "")),
                            "link": e.get("link", ""),
                        })
                    except Exception:
                        continue
            except Exception as ex:
                logger.warning(f"RSS failed {url}: {ex}")
        # Google News per tennisti top (best-effort, 1 query generica)
        try:
            q = quote_plus("tennis ATP WTA injury withdraw")
            r = requests.get(
                f"https://news.google.com/rss/search?q={q}&hl=it&gl=IT&ceid=IT:it",
                timeout=20, headers={"User-Agent": "Mozilla/5.0"},
            )
            if r.ok:
                for e in feedparser.parse(r.text).entries[:30]:
                    articles.append({
                        "title": e.get("title", ""), "text": e.get("title", ""),
                        "published": str(e.get("published", "")),
                        "link": e.get("link", ""),
                    })
        except Exception as ex:
            logger.warning(f"Google News failed: {ex}")
        logger.info(f"News: {len(articles)} articoli (ultime {hours_back}h)")
        return articles

    def _score_text(self, text: str) -> float:
        t = _norm(text)
        score = 0.0
        for pat, w in NEG_PATTERNS + POS_PATTERNS:
            if re.search(pat, t):
                score += w
        return max(-1.0, min(1.0, score))

    def fetch_player_news(self, player: str, limit: int = 8) -> list[dict]:
        """Google News RSS dedicato al singolo giocatore (con cache per run)."""
        key = _norm(player)
        if key in self._player_cache:
            return self._player_cache[key]
        out: list[dict] = []
        try:
            q = quote_plus(f'"{player}" tennis')
            r = requests.get(
                f"https://news.google.com/rss/search?q={q}&hl=it&gl=IT&ceid=IT:it",
                timeout=15, headers={"User-Agent": "Mozilla/5.0"},
            )
            if r.ok:
                for e in feedparser.parse(r.text).entries[:limit]:
                    out.append({
                        "title": e.get("title", ""), "text": e.get("title", ""),
                        "published": str(e.get("published", "")),
                        "link": e.get("link", ""),
                    })
        except Exception as ex:
            logger.warning(f"Google News '{player}' failed: {ex}")
        self._player_cache[key] = out
        return out

    def get_player_news_impact(self, player: str, hours_back: int = 72,
                               articles: list[dict] = None) -> dict:
        arts = list(articles) if articles is not None else self.fetch_all_feeds(hours_back)
        # scansione dedicata per giocatore (infortuni, malattie, ritiri...)
        try:
            arts.extend(self.fetch_player_news(player))
        except Exception:
            pass
        toks = [t for t in _norm(player).split() if len(t) > 2]
        if not toks:
            return {"impact": 0.0, "confidence": 0.0, "article_count": 0}
        hits = []
        for a in arts:
            t = _norm(f"{a.get('title', '')} {a.get('text', '')}")
            if sum(1 for tk in toks if tk in t) >= (1 if len(toks) == 1 else 2):
                hits.append(a)
        if not hits:
            return {"impact": 0.0, "confidence": 0.0, "article_count": 0}
        scores = [self._score_text(f"{h.get('title', '')} {h.get('text', '')}")
                  for h in hits]
        impact = sum(scores) / len(scores)
        return {
            "impact": round(float(impact), 3),
            "confidence": round(min(1.0, len(hits) / 3.0), 3),
            "article_count": len(hits),
            "headlines": [h.get("title", "")[:120] for h in hits[:3]],
        }
