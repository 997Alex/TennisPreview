"""Baseline models for tennis prediction"""
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime, timedelta

from src.utils.config import config
from src.utils.logging import get_logger
from src.utils.models import TennisFeatures, TennisPrediction, DataTier, Surface, Player

logger = get_logger("tennis_baseline_models")


class BradleyTerryModel:
    """Bradley-Terry model for head-to-head win probability"""
    
    def __init__(self, surface_specific: bool = True):
        self.surface_specific = surface_specific
        self.ratings = defaultdict(lambda: defaultdict(float))  # player -> surface -> rating
        self.surface_default = 1500.0
        self.k_factor = 32
    
    @staticmethod
    def _parse_surface_safe(value) -> Surface:
        try:
            if isinstance(value, Surface):
                return value
            if value is None or (isinstance(value, float) and str(value) == 'nan'):
                return Surface.HARD
            s = str(value).strip()
            # Sackmann uses 'Hard', 'Clay', 'Grass', 'Carpet'
            for surf in Surface:
                if surf.value.lower() == s.lower():
                    return surf
            # Common aliases
            low = s.lower()
            if 'indoor' in low or low == 'i':
                return Surface.HARD_INDOOR
            if low.startswith('hard'):
                return Surface.HARD
            if low.startswith('clay'):
                return Surface.CLAY
            if low.startswith('grass'):
                return Surface.GRASS
            if low.startswith('carpet'):
                return Surface.CARPET
            return Surface.HARD
        except Exception:
            return Surface.HARD

    def fit(self, matches: List[Dict], players: Dict[str, Player]):
        """Fit model on historical matches"""
        logger.info(f"Fitting Bradley-Terry on {len(matches)} matches")

        # Initialize ratings from player data
        for pid, player in players.items():
            for surface in Surface:
                if surface in player.surface_elo:
                    self.ratings[str(pid)][surface] = player.surface_elo[surface]
                else:
                    self.ratings[str(pid)][surface] = self.surface_default

        # Sort matches by date
        matches_sorted = sorted(
            matches,
            key=lambda m: str(m.get('tourney_date', m.get('date', '')))
        )

        for match in matches_sorted:
            winner = match.get('winner_id')
            loser = match.get('loser_id')
            surface = self._parse_surface_safe(match.get('surface', 'Hard'))
            
            if winner is None or loser is None:
                continue
            
            winner = str(winner)
            loser = str(loser)
            
            r_w = self.ratings[winner][surface]
            r_l = self.ratings[loser][surface]
            
            # Expected score
            e_w = 1 / (1 + 10 ** ((r_l - r_w) / 400))
            e_l = 1 - e_w
            
            # Update
            self.ratings[winner][surface] += self.k_factor * (1 - e_w)
            self.ratings[loser][surface] += self.k_factor * (0 - e_l)
        
        logger.info(f"Bradley-Terry fitted for {len(self.ratings)} players")
    
    def predict(self, player1_id: str, player2_id: str, 
                surface: Surface) -> float:
        """Predict P(player1 beats player2)"""
        r1 = self.ratings.get(str(player1_id), {}).get(surface, self.surface_default)
        r2 = self.ratings.get(str(player2_id), {}).get(surface, self.surface_default)
        
        return 1 / (1 + 10 ** ((r2 - r1) / 400))
    
    def get_rating(self, player_id: str, surface: Surface) -> float:
        return self.ratings.get(str(player_id), {}).get(surface, self.surface_default)


class Glicko2Model:
    """Glicko-2 rating system for tennis (surface-specific)"""
    
    def __init__(self, tau: float = 0.5):
        self.tau = tau  # Volatility constraint
        self.ratings = {}  # player -> {surface: {rating, rd, vol}}
        self.default_rating = 1500.0
        self.default_rd = 350.0
        self.default_vol = 0.06
    
    def _g(self, rd: float) -> float:
        return 1 / np.sqrt(1 + 3 * rd**2 / np.pi**2)
    
    def _e(self, rating: float, opp_rating: float, opp_rd: float) -> float:
        return 1 / (1 + np.exp(-self._g(opp_rd) * (rating - opp_rating) / 400))
    
    def fit(self, matches: List[Dict], players: Dict[str, Player]):
        """Fit Glicko-2 on historical matches"""
        logger.info(f"Fitting Glicko-2 on {len(matches)} matches")
        
        # Initialize
        for pid, player in players.items():
            self.ratings[str(pid)] = {}
            for surface in Surface:
                if surface in player.surface_elo:
                    self.ratings[str(pid)][surface] = {
                        'rating': player.surface_elo[surface],
                        'rd': self.default_rd,
                        'vol': self.default_vol
                    }
                else:
                    self.ratings[str(pid)][surface] = {
                        'rating': self.default_rating,
                        'rd': self.default_rd,
                        'vol': self.default_vol
                    }
        
        # Sort matches by date string (Sackmann tourney_date is YYYYMMDD)
        def _date_key(m):
            d = m.get('tourney_date', m.get('date'))
            if d is None:
                return ''
            if hasattr(d, 'strftime'):
                return d.strftime('%Y%m%d')
            return str(d)
        
        matches_sorted = sorted(matches, key=_date_key)
        
        # Process in batches per month
        current_period = None
        period_matches = []

        for match in matches_sorted:
            d = _date_key(match)
            if not d or len(str(d)) < 6:
                continue

            period = str(d)[:6]  # YYYYMM

            if current_period is None:
                current_period = period

            if period != current_period:
                self._update_ratings(current_period, period_matches)
                period_matches = []
                current_period = period

            period_matches.append(match)
        
        # Final period
        if period_matches:
            self._update_ratings(current_period, period_matches)
        
        logger.info(f"Glicko-2 fitted for {len(self.ratings)} players")
    
    def _update_ratings(self, period: str, matches: List[Dict]):
        """Update ratings for a rating period"""
        # Group by player and surface
        player_results = defaultdict(lambda: defaultdict(list))
        
        for match in matches:
            winner = match.get('winner_id')
            loser = match.get('loser_id')
            surface = BradleyTerryModel._parse_surface_safe(match.get('surface', 'Hard'))

            if winner is None or loser is None:
                continue

            player_results[str(winner)][surface].append(('win', str(loser)))
            player_results[str(loser)][surface].append(('loss', str(winner)))
        
        # Update each player
        for pid, surfaces in player_results.items():
            if pid not in self.ratings:
                continue
            
            for surface, results in surfaces.items():
                if surface not in self.ratings[pid]:
                    continue
                
                player_state = self.ratings[pid][surface]
                rating = player_state['rating']
                rd = player_state['rd']
                vol = player_state['vol']
                
                # Glicko-2 update (simplified)
                if results:
                    v_inv = 0
                    delta_sum = 0
                    
                    for result, opp_id in results:
                        if opp_id not in self.ratings or surface not in self.ratings[opp_id]:
                            continue
                        
                        opp_state = self.ratings[opp_id][surface]
                        opp_rating = opp_state['rating']
                        opp_rd = opp_state['rd']
                        
                        g = self._g(opp_rd)
                        e = self._e(rating, opp_rating, opp_rd)
                        
                        v_inv += g**2 * e * (1 - e)
                        delta_sum += g * (1 if result == 'win' else 0 - e)
                    
                    if v_inv > 0:
                        v = 1 / v_inv
                        delta = v * delta_sum
                        
                        # Update volatility (simplified)
                        new_vol = vol  # Would need iterative solution
                        
                        # Update rating and RD
                        new_rd = np.sqrt(rd**2 + new_vol**2)
                        new_rating = rating + new_rd**2 * delta_sum
                        
                        # Post-period RD increase
                        new_rd = np.sqrt(new_rd**2 + new_vol**2)
                        
                        player_state['rating'] = new_rating
                        player_state['rd'] = min(new_rd, 350)
                        player_state['vol'] = new_vol
                else:
                    # No matches: increase RD
                    player_state['rd'] = min(np.sqrt(rd**2 + vol**2), 350)
    
    def predict(self, player1_id: str, player2_id: str, surface: Surface) -> float:
        """Predict win probability"""
        player1_id = str(player1_id)
        player2_id = str(player2_id)
        if player1_id not in self.ratings or surface not in self.ratings[player1_id]:
            return 0.5
        if player2_id not in self.ratings or surface not in self.ratings[player2_id]:
            return 0.5
        
        p1 = self.ratings[player1_id][surface]
        p2 = self.ratings[player2_id][surface]
        
        g = self._g(p2['rd'])
        return 1 / (1 + np.exp(-g * (p1['rating'] - p2['rating']) / 400))
    
    def get_rating(self, player_id: str, surface: Surface) -> float:
        player_id = str(player_id)
        if player_id in self.ratings and surface in self.ratings[player_id]:
            return self.ratings[player_id][surface]['rating']
        return self.default_rating


class EloFormModel:
    """Simple Elo + recent form model"""
    
    def __init__(self):
        self.elo_ratings = {}
        self.k_factor = 24
        self.form_weight = 0.3
    
    @staticmethod
    def _date_key(m: Dict):
        # Historical dicts use 'tourney_date' (YYYYMMDD int/str/datetime),
        # not 'date'. Handle all formats robustly.
        for k in ('tourney_date', 'date'):
            d = m.get(k)
            if d is None:
                continue
            if hasattr(d, 'strftime'):
                try:
                    return d.strftime('%Y%m%d')
                except Exception:
                    continue
            s = str(d).strip()
            # normalize '2024-01-01' / '20240101' / float timestamps
            digits = ''.join(ch for ch in s if ch.isdigit())
            if len(digits) >= 8:
                return digits[:8]
            if s:
                return s
        return '19000101'

    def fit(self, matches: List[Dict], players: Dict[str, Player]):
        self.elo_ratings = {}

        for pid, player in players.items():
            # Average surface Elo as base
            elos = list(player.surface_elo.values())
            self.elo_ratings[pid] = float(np.mean(elos)) if elos else 1500.0

        matches_sorted = sorted(matches, key=self._date_key)
        
        for match in matches_sorted:
            winner = match.get('winner_id')
            loser = match.get('loser_id')
            
            if not winner or not loser:
                continue
            
            r_w = self.elo_ratings.get(winner, 1500.0)
            r_l = self.elo_ratings.get(loser, 1500.0)
            
            e_w = 1 / (1 + 10 ** ((r_l - r_w) / 400))
            e_l = 1 - e_w
            
            self.elo_ratings[winner] = r_w + self.k_factor * (1 - e_w)
            self.elo_ratings[loser] = r_l + self.k_factor * (0 - e_l)
    
    def predict(self, features: TennisFeatures) -> float:
        """Predict using Elo + form"""
        p1_elo = features.p1_surface_elo
        p2_elo = features.p2_surface_elo
        
        # Elo probability
        elo_prob = 1 / (1 + 10 ** ((p2_elo - p1_elo) / 400))
        
        # Form adjustment
        form_diff = features.p1_form_10 - features.p2_form_10
        form_adj = form_diff * self.form_weight
        
        # Rest adjustment
        rest_diff = (features.p1_rest_days - features.p2_rest_days) / 14
        rest_adj = rest_diff * 0.02
        
        combined = elo_prob + form_adj + rest_adj
        return np.clip(combined, 0.05, 0.95)


def get_baseline_models() -> Dict:
    """Get all baseline models"""
    return {
        'bradley_terry': BradleyTerryModel(),
        'glicko2': Glicko2Model(),
        'elo_form': EloFormModel(),
    }