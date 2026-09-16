"""Flashscore scraper (best-effort via Playwright; mai bloccante)."""

from src.utils.logging import get_logger

logger = get_logger("flashscore")


class FlashscoreScraper:
    def __init__(self):
        self._browser = None

    async def get_today_matches(self) -> list:
        # Parsing Flashscore richiede JS pesante + anti-bot: per ora nessun
        # match (ESPN + Sisal coprono il palinsesto). Ritorna [] senza errori.
        return []

    async def close(self) -> None:
        return None
