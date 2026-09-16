"""API-Football client (stub: attivo solo con chiave valida)."""
import os

from src.utils.logging import get_logger

logger = get_logger("api_football")


class APIFootballClient:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("API_FOOTBALL_KEY", "")

    def get_fixtures(self, target_date) -> list:
        return []

    def get_odds(self, match_id: str):
        return []
