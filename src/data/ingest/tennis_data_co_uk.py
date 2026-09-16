"""Tennis-Data.co.uk ingester (stub: download best-effort, mai bloccante)."""
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("tennis_data_co_uk")


class TennisDataCoUkIngester:
    def __init__(self, download_dir: str = "data/tennis_data_co_uk"):
        self.download_dir = download_dir

    def download(self, *args, **kwargs) -> None:
        logger.info("tennis-data.co.uk download: skip (stub)")

    def load(self, *args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()
