"""Risk management engine for tennis betting"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.utils.config import config
from src.utils.logging import get_logger
from src.utils.models import ValueBet, ApprovedBet

logger = get_logger("tennis_risk_engine")


@dataclass
class ExposureState:
    """Current exposure state"""
    daily_exposure: float = 0.0
    weekly_exposure: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    active_bets: Dict[str, ApprovedBet] = None
    bets_today: int = 0
    bets_this_week: int = 0
    
    def __post_init__(self):
        if self.active_bets is None:
            self.active_bets = {}


class TennisRiskEngine:
    """Risk management with multi-layer controls"""
    
    def __init__(self, bankroll: float = 10000.0):
        self.config = config.get('risk')
        self.bankroll = bankroll
        
        # Risk limits
        self.max_daily_exposure = self.config.get('max_daily_exposure', 0.05)
        self.max_weekly_exposure = self.config.get('max_weekly_exposure', 0.15)
        self.max_bet_per_match = self.config.get('max_bet_per_match', 0.02)
        self.max_bets_per_day = self.config.get('max_bets_per_day', 10)
        self.daily_stop_loss = self.config.get('daily_stop_loss', 0.03)
        self.weekly_stop_loss = self.config.get('weekly_stop_loss', 0.10)
        self.correlation_limit = self.config.get('correlation_limit', 0.7)
        
        # Exposure tracking
        self.exposure = ExposureState()
        self.bet_history = []
    
    def approve_bets(self, value_bets: List[ValueBet], 
                     current_exposure: ExposureState = None) -> List[ApprovedBet]:
        """Approve and size bets based on risk rules"""
        
        if current_exposure:
            self.exposure = current_exposure
        
        # Check stop losses
        if self._check_stop_losses():
            logger.warning("Stop loss triggered - no new bets approved")
            return []
        
        # Check daily bet limit
        if self.exposure.bets_today >= self.max_bets_per_day:
            logger.warning(f"Daily bet limit reached ({self.max_bets_per_day})")
            return []
        
        approved = []
        remaining_slots = self.max_bets_per_day - self.exposure.bets_today
        
        # Sort by confidence * edge
        sorted_bets = sorted(value_bets, 
                           key=lambda b: b.confidence * b.edge, 
                           reverse=True)
        
        for bet in sorted_bets[:remaining_slots]:
            approved_bet = self._evaluate_bet(bet)
            if approved_bet and approved_bet.risk_checks_passed:
                approved.append(approved_bet)
                self._update_exposure(approved_bet)
        
        return approved
    
    def _evaluate_bet(self, bet: ValueBet) -> Optional[ApprovedBet]:
        """Evaluate single bet against risk rules"""
        
        # Calculate raw stake
        raw_stake = self.bankroll * bet.stake_fraction
        
        # Apply max bet per match limit
        max_stake = self.bankroll * self.max_bet_per_match
        stake = min(raw_stake, max_stake)
        
        # Check if stake would exceed daily exposure
        new_daily_exposure = self.exposure.daily_exposure + stake / self.bankroll
        if new_daily_exposure > self.max_daily_exposure:
            # Scale down
            max_additional = self.bankroll * (self.max_daily_exposure - self.exposure.daily_exposure)
            if max_additional < 1.0:  # Less than 1 unit
                logger.debug(f"Bet rejected: would exceed daily exposure limit")
                return ApprovedBet(
                    value_bet=bet,
                    stake=0,
                    bankroll=self.bankroll,
                    risk_checks_passed=False,
                    rejection_reason="daily_exposure_limit"
                )
            stake = max_additional
        
        # Check weekly exposure
        new_weekly_exposure = self.exposure.weekly_exposure + stake / self.bankroll
        if new_weekly_exposure > self.max_weekly_exposure:
            max_additional = self.bankroll * (self.max_weekly_exposure - self.exposure.weekly_exposure)
            if max_additional < 1.0:
                logger.debug(f"Bet rejected: would exceed weekly exposure limit")
                return ApprovedBet(
                    value_bet=bet,
                    stake=0,
                    bankroll=self.bankroll,
                    risk_checks_passed=False,
                    rejection_reason="weekly_exposure_limit"
                )
            stake = max_additional
        
        # Correlation check (same tournament, same player)
        if not self._check_correlation(bet):
            return ApprovedBet(
                value_bet=bet,
                stake=0,
                bankroll=self.bankroll,
                risk_checks_passed=False,
                rejection_reason="correlation_limit"
            )
        
        # Minimum stake check
        if stake < 1.0:
            return ApprovedBet(
                value_bet=bet,
                stake=0,
                bankroll=self.bankroll,
                risk_checks_passed=False,
                rejection_reason="stake_too_small"
            )
        
        return ApprovedBet(
            value_bet=bet,
            stake=round(stake, 2),
            bankroll=self.bankroll,
            risk_checks_passed=True
        )
    
    def _check_correlation(self, bet: ValueBet) -> bool:
        """Check correlation with existing bets"""
        # Simple checks: same tournament, same player
        for active_bet in self.exposure.active_bets.values():
            # Same player
            if active_bet.value_bet.player_name == bet.player_name:
                return False
            
            # Same tournament (approximate)
            if (active_bet.value_bet.tournament == bet.tournament and
                active_bet.value_bet.tournament != ''):
                return False
        
        return True
    
    def _check_stop_losses(self) -> bool:
        """Check if stop losses are triggered"""
        daily_loss_pct = abs(self.exposure.daily_pnl) / self.bankroll
        weekly_loss_pct = abs(self.exposure.weekly_pnl) / self.bankroll
        
        if self.exposure.daily_pnl < 0 and daily_loss_pct >= self.daily_stop_loss:
            logger.warning(f"Daily stop loss hit: {daily_loss_pct:.2%}")
            return True
        
        if self.exposure.weekly_pnl < 0 and weekly_loss_pct >= self.weekly_stop_loss:
            logger.warning(f"Weekly stop loss hit: {weekly_loss_pct:.2%}")
            return True
        
        return False
    
    def _update_exposure(self, bet: ApprovedBet):
        """Update exposure after bet approval"""
        self.exposure.daily_exposure += bet.stake / self.bankroll
        self.exposure.weekly_exposure += bet.stake / self.bankroll
        self.exposure.bets_today += 1
        self.exposure.bets_this_week += 1
        self.exposure.active_bets[bet.value_bet.match_id] = bet
    
    def settle_bet(self, match_id: str, won: bool, odds: float, stake: float):
        """Settle a bet and update P&L"""
        if match_id in self.exposure.active_bets:
            bet = self.exposure.active_bets.pop(match_id)
            
            if won:
                pnl = stake * (odds - 1)
            else:
                pnl = -stake
            
            self.exposure.daily_pnl += pnl
            self.exposure.weekly_pnl += pnl
            self.exposure.daily_exposure -= stake / self.bankroll
            self.exposure.weekly_exposure -= stake / self.bankroll
            
            self.bet_history.append({
                'match_id': match_id,
                'won': won,
                'stake': stake,
                'odds': odds,
                'pnl': pnl,
                'timestamp': datetime.now()
            })
            
            logger.info(f"Bet settled: {match_id} - {'WON' if won else 'LOST'} "
                       f"PnL: {pnl:.2f}")
    
    def reset_daily(self):
        """Reset daily counters"""
        self.exposure.daily_exposure = 0.0
        self.exposure.daily_pnl = 0.0
        self.exposure.bets_today = 0
        logger.info("Daily exposure reset")
    
    def reset_weekly(self):
        """Reset weekly counters"""
        self.exposure.weekly_exposure = 0.0
        self.exposure.weekly_pnl = 0.0
        self.exposure.bets_this_week = 0
        logger.info("Weekly exposure reset")
    
    def get_summary(self) -> Dict:
        """Get risk summary"""
        return {
            'bankroll': self.bankroll,
            'daily_exposure_pct': self.exposure.daily_exposure * 100,
            'weekly_exposure_pct': self.exposure.weekly_exposure * 100,
            'daily_pnl': self.exposure.daily_pnl,
            'weekly_pnl': self.exposure.weekly_pnl,
            'bets_today': self.exposure.bets_today,
            'bets_this_week': self.exposure.bets_this_week,
            'active_bets': len(self.exposure.active_bets),
            'daily_stop_loss_pct': self.daily_stop_loss * 100,
            'weekly_stop_loss_pct': self.weekly_stop_loss * 100,
        }


def get_risk_engine(bankroll: float = 10000.0) -> TennisRiskEngine:
    return TennisRiskEngine(bankroll)