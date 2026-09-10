"""Value bet detection for tennis"""
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.utils.config import config
from src.utils.logging import get_logger
from src.utils.models import TennisFeatures, TennisPrediction, MergedOdds, ValueBet, DataTier

logger = get_logger("tennis_value_detector")


def _side(feat_val_p1, feat_val_p2, side: str):
    """Pick (mine, theirs) oriented to the backed side."""
    if side == "player2":
        return feat_val_p2, feat_val_p1
    return feat_val_p1, feat_val_p2


def explain_value_bet(vb: ValueBet, feat: Optional[TennisFeatures] = None) -> str:
    """Human reason (Italian) why this side is favored over the other."""
    s = vb.side or "player1"
    parts: List[str] = []

    if feat is not None:
        surf = feat.surface.value if hasattr(feat.surface, "value") else str(feat.surface)
        e1, e2 = _side(feat.p1_surface_elo, feat.p2_surface_elo, s)
        parts.append(f"Elo {surf} {e1:.0f} vs {e2:.0f} ({e1 - e2:+.0f})")

        f1, f2 = _side(feat.p1_form_10, feat.p2_form_10, s)
        parts.append(f"forma {round(f1 * 10)}V-{10 - round(f1 * 10)}P "
                     f"vs {round(f2 * 10)}V-{10 - round(f2 * 10)}P (ultime 10)")

        if feat.h2h_total > 0:
            w, losses = _side(feat.h2h_p1_wins, feat.h2h_total - feat.h2h_p1_wins, s)
            h = f"H2H {w}-{losses}"
            if feat.h2h_surface > 0:
                ws, ls = _side(feat.h2h_p1_wins_surface,
                               feat.h2h_surface - feat.h2h_p1_wins_surface, s)
                h += f" ({ws}-{ls} su {surf})"
            parts.append(h)

        sv1, sv2 = _side(feat.p1_serve_rating, feat.p2_serve_rating, s)
        rt1, rt2 = _side(feat.p1_return_rating, feat.p2_return_rating, s)
        if abs(sv1 - sv2) >= 3:
            parts.append(f"battuta {sv1:.0f} vs {sv2:.0f}")
        if abs(rt1 - rt2) >= 3:
            parts.append(f"risposta {rt1:.0f} vs {rt2:.0f}")

        n1, n2 = _side(feat.p1_news_impact, feat.p2_news_impact, s)
        if abs(n1) >= 0.15 or abs(n2) >= 0.15:
            names = _side(vb.player1, vb.player2, s)
            bits = []
            if abs(n1) >= 0.15:
                bits.append(f"{names[0]}: {'positive' if n1 > 0 else 'negative'} ({n1:+.2f})")
            if abs(n2) >= 0.15:
                bits.append(f"{names[1]}: {'positive' if n2 > 0 else 'negative'} ({n2:+.2f})")
            parts.append("news " + ", ".join(bits))
        else:
            parts.append("news neutre")

        pside = getattr(feat, "partial_side", "")
        if pside:
            bare = []
            if pside in ("player1", "both"):
                bare.append(vb.player1)
            if pside in ("player2", "both"):
                bare.append(vb.player2)
            parts.append(f"{', '.join(bare)} senza storico (profilo parziale)")

    if feat is not None and getattr(feat, "poly_implied_p1", 0.0) > 0:
        pp1, pp2 = _side(feat.poly_implied_p1, feat.poly_implied_p2, s)
        parts.append(f"sentiment Polymarket {pp1:.1%} vs {pp2:.1%}")

    parts.append(f"mercato {vb.implied_prob:.1%} vs modello {vb.model_prob:.1%} "
                 f"(edge {vb.edge:+.1%})")
    return "; ".join(parts)


def sisal_coverage(tier) -> Tuple[str, str]:
    """Expected Sisal listing. Sisal.it (bot-protected, verified manually)
    lists Slams + ATP/WTA tour; WTA 125 / Challenger / ITF are not listed.
    Returns (flag, note): flag 'si'|'no'."""
    t = tier.value if hasattr(tier, "value") else tier
    if t == 1:
        return "si", ""
    return ("no",
            "fuori palinsesto Sisal (125/Challenger/ITF non quotati) - "
            "verificare su Bet365/Pinnacle/Unibet")


class TennisValueDetector:
    """Detect value bets with tier-aware thresholds"""
    
    def __init__(self):
        self.config = config.get('tennis', 'betting')
        self.tiers_config = config.get('tennis', 'tiers')
        
        # Build thresholds from tier config
        self.tier_thresholds = {}
        tier_key_map = {
            'tier_1': 1, 'tier_2': 2, 'tier_3': 3, 'tier_4': 4,
        }
        for tier_name, tier_int in tier_key_map.items():
            tier_cfg = self.tiers_config.get(tier_name, {})
            self.tier_thresholds[tier_int] = {
                'min_edge': tier_cfg.get('min_edge', 0.05),
                'min_prob': tier_cfg.get('min_prob', 0.1),
                'max_odds': tier_cfg.get('max_odds', 3.0),
                'kelly_frac': tier_cfg.get('kelly_fraction', 0.1),
            }
        
        # Default thresholds if not in config
        self.default_thresholds = {
            1: {'min_edge': 0.025, 'min_prob': 0.10, 'max_odds': 4.0, 'kelly_frac': 0.25},
            2: {'min_edge': 0.035, 'min_prob': 0.12, 'max_odds': 3.5, 'kelly_frac': 0.18},
            3: {'min_edge': 0.050, 'min_prob': 0.15, 'max_odds': 3.0, 'kelly_frac': 0.10},
            4: {'min_edge': 0.000, 'min_prob': 0.00, 'max_odds': 0.0, 'kelly_frac': 0.00},
        }
        
        self.min_clv = self.config.get('min_clv', 0.01)
        self.max_bets_per_tournament = self.config.get('max_bets_per_tournament_day', 3)
        self.max_bets_per_player_week = self.config.get('max_bets_per_player_week', 2)
    
    def find_value(self, pred: TennisPrediction, 
                   odds: MergedOdds,
                   match_info: dict = None) -> Optional[ValueBet]:
        """
        Find value bet for a match
        
        Returns best value bet or None if no value found
        """
        tier = pred.tier
        tier_int = tier.value if hasattr(tier, 'value') else tier
        
        thresholds = self.tier_thresholds.get(tier_int, 
                     self.default_thresholds.get(tier_int, self.default_thresholds[3]))
        
        # Tier 4: never bet
        if tier_int == 4:
            return None
        
        min_edge = thresholds['min_edge']
        min_prob = thresholds['min_prob']
        max_odds = thresholds['max_odds']
        kelly_frac = thresholds['kelly_frac']
        
        # Check odds availability
        if odds.best_odds_home <= 1.0 or odds.best_odds_away <= 1.0:
            return None
        
        # Calculate edges
        edge_home = pred.p1_win_prob - odds.implied_prob_home
        edge_away = pred.p2_win_prob - odds.implied_prob_away
        
        candidates = []
        
        # Check home (player1)
        if (edge_home > min_edge and 
            pred.p1_win_prob > min_prob and
            odds.best_odds_home < max_odds):
            
            kelly = edge_home / (odds.best_odds_home - 1)
            stake_frac = kelly * kelly_frac
            
            # CLV check
            clv_home = odds.clv_estimate.get('home', 0)
            
            candidates.append(ValueBet(
                match_id=pred.match_id,
                match_key=match_info.get('match_key', '') if match_info else '',
                tournament=match_info.get('tournament', '') if match_info else '',
                player1=match_info.get('player1', '') if match_info else '',
                player2=match_info.get('player2', '') if match_info else '',
                side='player1',
                player_name=match_info.get('player1', '') if match_info else '',
                odds=odds.best_odds_home,
                model_prob=pred.p1_win_prob,
                implied_prob=odds.implied_prob_home,
                edge=edge_home,
                kelly_fraction=kelly,
                stake_fraction=stake_frac,
                tier=tier,
                confidence=pred.confidence,
                data_quality=pred.data_quality_flag,
                clv_estimate=clv_home
            ))
        
        # Check away (player2)
        if (edge_away > min_edge and 
            pred.p2_win_prob > min_prob and
            odds.best_odds_away < max_odds):
            
            kelly = edge_away / (odds.best_odds_away - 1)
            stake_frac = kelly * kelly_frac
            
            clv_away = odds.clv_estimate.get('away', 0)
            
            candidates.append(ValueBet(
                match_id=pred.match_id,
                match_key=match_info.get('match_key', '') if match_info else '',
                tournament=match_info.get('tournament', '') if match_info else '',
                player1=match_info.get('player1', '') if match_info else '',
                player2=match_info.get('player2', '') if match_info else '',
                side='player2',
                player_name=match_info.get('player2', '') if match_info else '',
                odds=odds.best_odds_away,
                model_prob=pred.p2_win_prob,
                implied_prob=odds.implied_prob_away,
                edge=edge_away,
                kelly_fraction=kelly,
                stake_fraction=stake_frac,
                tier=tier,
                confidence=pred.confidence,
                data_quality=pred.data_quality_flag,
                clv_estimate=clv_away
            ))
        
        if not candidates:
            return None
        
        # Return best by edge
        best = max(candidates, key=lambda b: b.edge)
        
        # Additional filters
        if best.clv_estimate < self.min_clv and best.clv_estimate != 0:
            logger.debug(f"Bet filtered: CLV {best.clv_estimate:.3f} < {self.min_clv}")
            return None
        
        if best.odds > max_odds:
            return None
        
        logger.info(f"Value bet found: {best.player_name} @ {best.odds:.2f} "
                   f"(edge: {best.edge:.2%}, tier: {tier_int})")
        
        return best
    
    def find_all_value(self, predictions: List[TennisPrediction],
                       odds_dict: Dict[str, MergedOdds],
                       matches_info: Dict[str, dict]) -> List[ValueBet]:
        """Find value bets for multiple matches"""
        value_bets = []
        
        for pred in predictions:
            odds = odds_dict.get(pred.match_id)
            if not odds:
                continue
            
            match_info = matches_info.get(pred.match_id, {})
            
            bet = self.find_value(pred, odds, match_info)
            if bet:
                value_bets.append(bet)
        
        # Apply tournament/player limits
        value_bets = self._apply_limits(value_bets)
        
        return value_bets
    
    def _apply_limits(self, bets: List[ValueBet]) -> List[ValueBet]:
        """Apply per-tournament and per-player limits"""
        # Count bets per tournament
        tournament_counts = {}
        player_counts = {}
        filtered = []
        
        # Sort by edge descending
        bets_sorted = sorted(bets, key=lambda b: b.edge, reverse=True)
        
        for bet in bets_sorted:
            tourney_key = f"{bet.tournament}_{bet.match_id[:8]}"  # Approximate
            player_key = bet.player_name
            
            tourney_count = tournament_counts.get(tourney_key, 0)
            player_count = player_counts.get(player_key, 0)
            
            if (tourney_count < self.max_bets_per_tournament and
                player_count < self.max_bets_per_player_week):
                filtered.append(bet)
                tournament_counts[tourney_key] = tourney_count + 1
                player_counts[player_key] = player_count + 1
            else:
                logger.debug(f"Bet filtered: limit reached for {tourney_key}/{player_key}")
        
        return filtered


def get_value_detector() -> TennisValueDetector:
    return TennisValueDetector()