"""Sisal.it tennis palinsesto scraper (Playwright, best-effort).

Sisal.it e' protetto da anti-bot: lo scraper apre Chromium headless,
attende il palinsesto tennis e estrae (torneo, giocatori, quote 1/2, orario).
Tutto e' cachato in data/sisal_today.json cosi' le esecuzioni successive
funzionano anche se Sisal blocca il browser.

Se lo scraping fallisce (blocco anti-bot), ritorna la cache del giorno
o [] senza mai crashare la pipeline.
"""
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger("sisal")

CACHE_PATH = Path("data/sisal_today.json")
SISAL_URLS = [
    "https://www.sisal.it/scommesse/tennis",
    "https://www.sisal.it/scommesse-matchpoint/palinsesto/tennis",
]

ODDS_RE = re.compile(r"^\d{1,2}[.,]\d{2}$")


@dataclass
class SisalMatch:
    id: str
    tournament: str = "Sisal Tennis"
    player1: str = ""
    player2: str = ""
    odds1: float = 0.0
    odds2: float = 0.0
    start_time: datetime | None = None
    extra: dict = field(default_factory=dict)


def _to_float_odds(txt: str) -> float:
    try:
        v = float(txt.replace(",", ".").strip())
        return v if 1.01 <= v <= 100.0 else 0.0
    except Exception:
        return 0.0


def save_cache(matches: list[SisalMatch], target: date) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps({
            "date": target.isoformat(),
            "matches": [
                {"id": m.id, "tournament": m.tournament, "player1": m.player1,
                 "player2": m.player2, "odds1": m.odds1, "odds2": m.odds2,
                 "start_time": m.start_time.isoformat() if m.start_time else None}
                for m in matches
            ],
        }, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Sisal cache save failed: {e}")


def load_cache(target: date) -> list[SisalMatch]:
    try:
        if not CACHE_PATH.exists():
            return []
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if data.get("date") != target.isoformat():
            return []
        out = []
        for d in data.get("matches", []):
            try:
                st = None
                if d.get("start_time"):
                    st = datetime.fromisoformat(d["start_time"])
                out.append(SisalMatch(
                    id=str(d.get("id")), tournament=str(d.get("tournament", "")),
                    player1=str(d.get("player1", "")), player2=str(d.get("player2", "")),
                    odds1=float(d.get("odds1") or 0), odds2=float(d.get("odds2") or 0),
                    start_time=st))
            except Exception:
                continue
        return out
    except Exception:
        return []


class SisalScraper:
    def __init__(self, headless=None, timeout_ms: int = 45000):
        import os as _os
        if headless is None:
            headless = _os.getenv("SISAL_HEADLESS", "1") != "0"
        self.headless = headless
        self.timeout_ms = timeout_ms

    async def get_matches_for_date(self, target: date) -> list[SisalMatch]:
        matches = await self._scrape_live()
        if not matches:
            matches = self._fetch_http()
            if matches:
                logger.info(f"Sisal: {len(matches)} match via HTTP fallback")
        if matches:
            save_cache(matches, target)
            logger.info(f"Sisal: {len(matches)} match live estratti")
            return matches
        cached = load_cache(target)
        if cached:
            logger.info(f"Sisal: uso cache ({len(cached)} match)")
            return cached
        logger.warning("Sisal: nessun match (blocco anti-bot? vedi cache)")
        return []

    async def _scrape_live(self) -> list[SisalMatch]:
        try:
            from playwright.async_api import async_playwright
        except Exception as e:
            logger.warning(f"Playwright non disponibile: {e}")
            return []
        out: list[SisalMatch] = []
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=self.headless)
                ctx = await browser.new_context(
                    locale="it-IT", timezone_id="Europe/Rome",
                    user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/126.0 Safari/537.36"),
                )
                page = await ctx.new_page()
                loaded = False
                for url in SISAL_URLS:
                    try:
                        await page.goto(url, timeout=self.timeout_ms,
                                        wait_until="domcontentloaded")
                        await page.wait_for_timeout(6000)
                        loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Sisal goto {url} failed: {e}")
                if not loaded:
                    await browser.close()
                    return []
                # Cookie banner best-effort
                for sel in ("button:has-text('Accetta')", "button:has-text('OK')",
                            "#onetrust-accept-btn-handler"):
                    try:
                        await page.click(sel, timeout=2500)
                        break
                    except Exception:
                        continue
                await page.wait_for_timeout(3000)
                text = await page.inner_text("body") or ""
                out = self._parse_text_blocks(text)
                if not out:
                    out = await self._parse_dom(page)
                await browser.close()
        except Exception as e:
            logger.warning(f"Sisal scrape failed: {e}")
            return []
        # dedup per coppia normalizzata
        seen, uniq = set(), []
        for m in out:
            key = (m.player1.lower().strip(), m.player2.lower().strip())
            if key[0] and key[1] and key not in seen:
                seen.add(key)
                uniq.append(m)
        return uniq

    @staticmethod
    def _parse_text_blocks(text: str) -> list[SisalMatch]:
        """Euristica su testo pagina: 'Giocatore A 1.85 Giocatore B 2.10'."""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        out: list[SisalMatch] = []
        i = 0
        while i < len(lines) - 3:
            a, b, c, d = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
            o1, o2 = _to_float_odds(b), _to_float_odds(d)

            def is_name(s: str) -> bool:
                return len(s) >= 3 and not ODDS_RE.match(s) and not any(
                    k in s for k in ("Sisal", "Bonus", "Cookie", "Accedi", "Registrati"))
            if is_name(a) and o1 and is_name(c) and o2:
                out.append(SisalMatch(
                    id=f"sisal_{abs(hash(a + c)) % 10**8}",
                    player1=a, player2=c, odds1=o1, odds2=o2))
                i += 4
            else:
                i += 1
        return out

    def _fetch_http(self) -> list[SisalMatch]:
        """Fallback: GET con fingerprint TLS Chrome (curl-cffi)."""
        try:
            from curl_cffi import requests as cr
        except Exception as e:
            logger.warning(f"curl-cffi non disponibile: {e}")
            return []
        out: list[SisalMatch] = []
        for url in SISAL_URLS:
            try:
                r = cr.get(url, impersonate="chrome124", timeout=30,
                           headers={"Accept-Language": "it-IT,it;q=0.9"})
                if r.status_code != 200:
                    continue
                html = r.text
                # spoglia tag -> righe di testo
                txt = re.sub(r"<script.*?</script>", "\n", html,
                             flags=re.S | re.I)
                txt = re.sub(r"<style.*?</style>", "\n", txt, flags=re.S | re.I)
                txt = re.sub(r"<[^>]+>", "\n", txt)
                txt = re.sub(r"\n+", "\n", txt)
                found = self._parse_text_blocks(txt)
                if found:
                    out.extend(found)
                    break
            except Exception as e:
                logger.warning(f"Sisal HTTP {url} failed: {e}")
        seen, uniq = set(), []
        for m in out:
            key = (m.player1.lower().strip(), m.player2.lower().strip())
            if key[0] and key[1] and key not in seen:
                seen.add(key)
                uniq.append(m)
        return uniq

    @staticmethod
    async def _parse_dom(page) -> list[SisalMatch]:
        out: list[SisalMatch] = []
        try:
            buttons = await page.query_selector_all("button")
            odds_btns = []
            for b in buttons:
                try:
                    t = (await b.inner_text()).strip()
                except Exception:
                    continue
                if ODDS_RE.match(t):
                    odds_btns.append(t)
            # senza nomi affidabili dal DOM generico, non inventiamo match:
            logger.info(f"Sisal DOM: {len(odds_btns)} bottoni quota, nomi non affidabili")
        except Exception as e:
            logger.warning(f"Sisal DOM parse failed: {e}")
        return out
