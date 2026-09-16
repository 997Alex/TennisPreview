"""Tennis feature builder with tiered data quality"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.utils.config import config
from src.utils.logging import get_logger
from src.utils.models import (
    TennisMatch, TennisFeatures, Player, Surface, 
    TournamentLevel, DataTier, OddsSnapshot
)
from src.data.merge.odds_merger import MergedOdds

logger = get_logger("tennis_features")


class TennisFeatureBuilder:
    """Build feature vectors for tennis match prediction with data quality tiers"""
    
    def __init__(self):
        self.config = config.get('tennis')
        self.tier_config = self.config.get('tiers', {})
        
        # Surface mapping
        self.surface_map = {
            'Hard': Surface.HARD,
            'Clay': Surface.CLAY,
            'Grass': Surface.GRASS,
            'Carpet': Surface.CARPET,
            'Hard (Indoor)': Surface.HARD_INDOOR,
        }
        
        # Tournament level to tier mapping
        self.tournament_tiers = {}
        for tier_name, tier_cfg in self.tier_config.items():
            if tier_name.startswith('tier_'):
                for t in tier_cfg.get('tournaments', []):
                    self.tournament_tiers[t] = tier_name
    
    def determine_tier(self, match: TennisMatch) -> DataTier:
        """Determine data quality tier for a match"""
        level_str = match.tournament_level.name
        
        tier_name = self.tournament_tiers.get(level_str, 'tier_3')
        
        tier_map = {
            'tier_1': DataTier.TIER_1,
            'tier_2': DataTier.TIER_2,
            'tier_3': DataTier.TIER_3,
            'tier_4': DataTier.TIER_4,
        }
        
        return tier_map.get(tier_name, DataTier.TIER_3)
    
    def build_features(self, match: TennisMatch,
                       player_db: Dict[str, Player],
                       merged_odds: Optional[MergedOdds] = None,
                       news_impact: Dict[str, Dict] = None,
                       h2h_data: List[Dict] = None,
                       as_of: datetime = None,
                       poly: Optional[tuple] = None) -> TennisFeatures:
        """Build complete feature vector for a match"""
        
        as_of = as_of or datetime.now()
        tier = self.determine_tier(match)
        
        # Get players - per-side fallback: a debutant without history keeps
        # his bare Player (all defaults: Elo 1500, win% 0.5, form 0.5) while
        # the known side keeps its real stats. Only when BOTH sides are
        # missing do we return empty features. The match is never dropped.
        p1 = player_db.get(match.player1.id) if match.player1 else None
        p2 = player_db.get(match.player2.id) if match.player2 else None

        p1_bare = p1 is None and match.player1 is not None
        p2_bare = p2 is None and match.player2 is not None
        if p1_bare:
            p1 = match.player1
            logger.info(f"Match {match.id}: '{p1.name}' senza storico -> default")
        if p2_bare:
            p2 = match.player2
            logger.info(f"Match {match.id}: '{p2.name}' senza storico -> default")

        if p1 is None or p2 is None:
            logger.warning(f"Missing player data for match {match.id}")
            return self._empty_features(match.id, tier)

        partial = p1_bare or p2_bare
        partial_side = ("both" if (p1_bare and p2_bare)
                        else "player1" if p1_bare
                        else "player2" if p2_bare else "")
        
        # Base features
        features = TennisFeatures(
            match_id=match.id,
            player1_id=p1.id,
            player2_id=p2.id,
            surface=match.surface,
            round=match.round,
            best_of=match.best_of,
            tournament_level=match.tournament_level,
            data_tier=tier
        )
        
        # Best available odds for all tiers
        if merged_odds:
            features.best_odds_home = merged_odds.best_odds_home
            features.best_odds_away = merged_odds.best_odds_away

        # Polymarket sentiment (0.0 = assente, come per il mercato)
        if poly:
            try:
                pp1, pp2 = float(poly[0]), float(poly[1])
                if 0.0 < pp1 < 1.0 and 0.0 < pp2 < 1.0:
                    features.poly_implied_p1 = pp1
                    features.poly_implied_p2 = pp2
            except (TypeError, ValueError, IndexError):
                pass
        
        # Fill features based on tier
        if tier in [DataTier.TIER_1, DataTier.TIER_2]:
            features = self._enrich_tier_1_2(features, p1, p2, match, merged_odds, news_impact, h2h_data, as_of)
        elif tier == DataTier.TIER_3:
            features = self._enrich_tier_3(features, p1, p2, match, merged_odds, h2h_data, as_of)
        else:
            features = self._enrich_tier_4(features, match, merged_odds)
        
        # News impact (all tiers if available)
        if news_impact:
            p1_news = news_impact.get(p1.name, {})
            p2_news = news_impact.get(p2.name, {})
            features.p1_news_impact = p1_news.get('impact', 0.0)
            features.p2_news_impact = p2_news.get('impact', 0.0)
        
        # Calculate completeness
        features.feature_completeness = self._calculate_completeness(features, tier)
        features.partial_profile = partial
        features.partial_side = partial_side

        return features
    
    def _enrich_tier_1_2(self, features: TennisFeatures, p1: Player, p2: Player,
                         match: TennisMatch, merged_odds: Optional[MergedOdds],
                         news_impact: Dict, h2h_data: List[Dict], as_of: datetime) -> TennisFeatures:
        """Enrich with full feature set for Tier 1-2"""
        
        surface = match.surface
        
        # Surface Elo
        features.p1_surface_elo = p1.get_surface_elo(surface)
        features.p2_surface_elo = p2.get_surface_elo(surface)
        
        # Surface win%
        features.p1_surface_win_pct = p1.surface_win_pct.get(surface, 0.5)
        features.p2_surface_win_pct = p2.surface_win_pct.get(surface, 0.5)
        
        # Serve/Return ratings
        features.p1_serve_rating = p1.get_serve_rating(surface)
        features.p2_serve_rating = p2.get_serve_rating(surface)
        features.p1_return_rating = p1.get_return_rating(surface)
        features.p2_return_rating = p2.get_return_rating(surface)
        
        # Recent form
        features.p1_matches_14d = p1.matches_last_14d
        features.p2_matches_14d = p2.matches_last_14d
        features.p1_sets_14d = p1.sets_last_14d
        features.p2_sets_14d = p2.sets_last_14d
        
        total1 = p1.wins_last_10 + p1.losses_last_10
        total2 = p2.wins_last_10 + p2.losses_last_10
        features.p1_form_10 = p1.wins_last_10 / total1 if total1 > 0 else 0.5
        features.p2_form_10 = p2.wins_last_10 / total2 if total2 > 0 else 0.5
        
        # H2H
        if h2h_data:
            features.h2h_total = len(h2h_data)
            features.h2h_p1_wins = sum(1 for m in h2h_data if m.get('winner') == p1.name)
            
            surface_h2h = [m for m in h2h_data if m.get('surface') == surface.value]
            features.h2h_surface = len(surface_h2h)
            features.h2h_p1_wins_surface = sum(1 for m in surface_h2h if m.get('winner') == p1.name)
        
        # Fitness
        features.p1_rest_days = p1.days_since_last_match
        features.p2_rest_days = p2.days_since_last_match
        features.p1_travel_km = p1.distance_travelled_km
        features.p2_travel_km = p2.distance_travelled_km
        features.p1_timezone_diff = p1.timezone_diff_hours
        features.p2_timezone_diff = p2.timezone_diff_hours
        
        # Market
        if merged_odds:
            features.market_implied_p1 = merged_odds.implied_prob_home
            features.market_implied_p2 = merged_odds.implied_prob_away
            features.odds_movement_p1 = 0.0  # Would need historical odds
            features.odds_movement_p2 = 0.0
        
        return features
    
    def _enrich_tier_3(self, features: TennisFeatures, p1: Player, p2: Player,
                       match: TennisMatch, merged_odds: Optional[MergedOdds],
                       h2h_data: List[Dict], as_of: datetime) -> TennisFeatures:
        """Enrich with limited features for Tier 3"""
        
        surface = match.surface
        
        # Basic surface win%
        features.p1_surface_win_pct = p1.surface_win_pct.get(surface, 0.5)
        features.p2_surface_win_pct = p2.surface_win_pct.get(surface, 0.5)
        
        # Approximate Elo from win%
        features.p1_surface_elo = 1500 + (features.p1_surface_win_pct - 0.5) * 400
        features.p2_surface_elo = 1500 + (features.p2_surface_win_pct - 0.5) * 400
        
        # Basic serve/return (default)
        features.p1_serve_rating = 50.0
        features.p2_serve_rating = 50.0
        features.p1_return_rating = 50.0
        features.p2_return_rating = 50.0
        
        # Recent form
        total1 = p1.wins_last_10 + p1.losses_last_10
        total2 = p2.wins_last_10 + p2.losses_last_10
        features.p1_form_10 = p1.wins_last_10 / total1 if total1 > 0 else 0.5
        features.p2_form_10 = p2.wins_last_10 / total2 if total2 > 0 else 0.5
        
        # H2H
        if h2h_data:
            features.h2h_total = len(h2h_data)
            features.h2h_p1_wins = sum(1 for m in h2h_data if m.get('winner') == p1.name)
            surface_h2h = [m for m in h2h_data if m.get('surface') == surface.value]
            features.h2h_surface = len(surface_h2h)
            features.h2h_p1_wins_surface = sum(1 for m in surface_h2h if m.get('winner') == p1.name)
        
        # Fitness
        features.p1_rest_days = p1.days_since_last_match
        features.p2_rest_days = p2.days_since_last_match
        
        # Market
        if merged_odds:
            features.market_implied_p1 = merged_odds.implied_prob_home
            features.market_implied_p2 = merged_odds.implied_prob_away
        
        return features
    
    def _enrich_tier_4(self, features: TennisFeatures, match: TennisMatch,
                       merged_odds: Optional[MergedOdds]) -> TennisFeatures:
        """Tier 4: Market only, no model features"""
        
        if merged_odds:
            features.market_implied_p1 = merged_odds.implied_prob_home
            features.market_implied_p2 = merged_odds.implied_prob_away
            features.best_odds_home = merged_odds.best_odds_home
            features.best_odds_away = merged_odds.best_odds_away
        
        return features
    
    def _empty_features(self, match_id: str, tier: DataTier) -> TennisFeatures:
        """Return empty features for missing data"""
        return TennisFeatures(
            match_id=match_id,
            data_tier=tier,
            feature_completeness=0.0
        )
    
    def _calculate_completeness(self, features: TennisFeatures, tier: DataTier) -> float:
        """Calculate feature completeness score"""
        if tier == DataTier.TIER_4:
            return 0.05
        elif tier == DataTier.TIER_3:
            # Check how many non-default features we have
            count = 0
            total = 0
            for name in features.feature_names:
                val = getattr(features, name)
                if val != 0.0 and val != 0.5 and val != 1500.0 and val != 50.0:
                    count += 1
                total += 1
            return 0.3 + 0.3 * (count / total)
        else:
            # Tier 1-2: check for serve/return ratings
            has_advanced = (features.p1_serve_rating != 50.0 or 
                           features.p1_return_rating != 50.0)
            base = 0.7
            if has_advanced:
                base = 0.9
            return min(1.0, base + features.h2h_total * 0.02)


def get_feature_builder() -> TennisFeatureBuilder:
    return TennisFeatureBuilder()