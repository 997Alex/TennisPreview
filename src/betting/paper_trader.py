"""Paper trading logger for tennis bets"""
import json
import sqlite3
from datetime import datetime, date
from typing import List, Dict, Optional
from pathlib import Path

from src.utils.config import config
from src.utils.logging import get_logger
from src.utils.models import ValueBet, ApprovedBet, TennisPrediction, MergedOdds

logger = get_logger("paper_trader")


class PaperTradeLogger:
    """Log and track paper trading bets"""
    
    def __init__(self, bankroll: float = 10000.0):
        self.config = config.get('app')
        self.output_dir = Path(self.config.get('output_dir', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.bankroll = bankroll
        self.current_bankroll = bankroll
        
        # SQLite database for persistence
        self.db_path = self.output_dir / "paper_trades.db"
        self._init_db()
        
        # Daily tracking
        self.daily_bets = []
        self.daily_pnl = 0.0
        self.bets_today = 0
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                match_key TEXT,
                tournament TEXT,
                player1 TEXT,
                player2 TEXT,
                side TEXT,
                player_name TEXT,
                odds REAL,
                model_prob REAL,
                implied_prob REAL,
                edge REAL,
                kelly_fraction REAL,
                stake_fraction REAL,
                stake REAL,
                tier INTEGER,
                confidence REAL,
                data_quality TEXT,
                clv_estimate REAL,
                status TEXT DEFAULT 'pending',
                result INTEGER,
                pnl REAL,
                placed_at TIMESTAMP,
                settled_at TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                date DATE PRIMARY KEY,
                starting_bankroll REAL,
                ending_bankroll REAL,
                total_staked REAL,
                total_pnl REAL,
                bets_count INTEGER,
                wins INTEGER,
                losses INTEGER,
                roi REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_bet(self, bet: ApprovedBet, prediction: TennisPrediction = None,
                odds: MergedOdds = None):
        """Log a new bet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        vb = bet.value_bet
        
        cursor.execute('''
            INSERT INTO bets (
                match_id, match_key, tournament, player1, player2,
                side, player_name, odds, model_prob, implied_prob,
                edge, kelly_fraction, stake_fraction, stake,
                tier, confidence, data_quality, clv_estimate,
                status, placed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            vb.match_id, vb.match_key, vb.tournament, vb.player1, vb.player2,
            vb.side, vb.player_name, vb.odds, vb.model_prob, vb.implied_prob,
            vb.edge, vb.kelly_fraction, vb.stake_fraction, bet.stake,
            vb.tier.value if hasattr(vb.tier, 'value') else vb.tier,
            vb.confidence, vb.data_quality, vb.clv_estimate,
            'pending', datetime.now().isoformat()
        ))
        
        bet_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        self.daily_bets.append({
            'id': bet_id,
            'bet': bet,
            'prediction': prediction,
            'odds': odds
        })
        self.bets_today += 1
        
        logger.info(f"Logged bet #{bet_id}: {vb.player_name} @ {vb.odds:.2f} "
                   f"Stake: {bet.stake:.2f}")
    
    def settle_bet(self, match_id: str, won: bool, final_odds: float = None):
        """Settle a bet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, stake, odds FROM bets WHERE match_id = ? AND status = ?',
                      (match_id, 'pending'))
        row = cursor.fetchone()
        
        if not row:
            logger.warning(f"No pending bet found for {match_id}")
            conn.close()
            return
        
        bet_id, stake, odds = row
        used_odds = final_odds or odds
        
        if won:
            pnl = stake * (used_odds - 1)
            result = 1
        else:
            pnl = -stake
            result = 0
        
        self.current_bankroll += pnl
        self.daily_pnl += pnl
        
        cursor.execute('''
            UPDATE bets SET status = ?, result = ?, pnl = ?, settled_at = ?
            WHERE id = ?
        ''', ('settled', result, pnl, datetime.now().isoformat(), bet_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Settled bet #{bet_id}: {'WON' if won else 'LOST'} "
                   f"PnL: {pnl:.2f} Bankroll: {self.current_bankroll:.2f}")
    
    def get_current_exposure(self) -> Dict:
        """Get current exposure state"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT SUM(stake) FROM bets WHERE status = ?
        ''', ('pending',))
        total_staked = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_staked': total_staked,
            'exposure_pct': total_staked / self.current_bankroll if self.current_bankroll > 0 else 0,
            'current_bankroll': self.current_bankroll,
            'daily_pnl': self.daily_pnl,
            'bets_today': self.bets_today
        }
    
    def generate_daily_report(self, report_date: date = None) -> Dict:
        """Generate daily performance report"""
        report_date = report_date or date.today()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get today's bets
        cursor.execute('''
            SELECT stake, result, pnl, odds, model_prob, edge, tier
            FROM bets 
            WHERE date(placed_at) = ? AND status = ?
        ''', (report_date.isoformat(), 'settled'))
        
        rows = cursor.fetchall()

        if not rows:
            conn.close()
            # Still persist an (empty) daily report so output/ always matches
            # README (daily_report_YYYY-MM-DD.json). Callers (pipeline) extend
            # it with analyzed_events / todays_bets afterwards.
            empty_report = {
                'date': report_date.isoformat(),
                'bets': 0,
                'message': 'No settled bets',
                'starting_bankroll': self.current_bankroll,
                'ending_bankroll': self.current_bankroll,
                'total_staked': 0,
                'total_pnl': 0,
                'roi': 0,
                'bets_count': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'tier_breakdown': {},
            }
            try:
                report_file = self.output_dir / f"daily_report_{report_date.isoformat()}.json"
                with open(report_file, 'w') as f:
                    json.dump(empty_report, f, indent=2)
            except Exception:
                pass
            return empty_report
        
        total_staked = sum(r[0] for r in rows)
        total_pnl = sum(r[2] for r in rows)
        wins = sum(1 for r in rows if r[1] == 1)
        losses = sum(1 for r in rows if r[1] == 0)
        
        roi = total_pnl / total_staked if total_staked > 0 else 0
        
        # By tier
        tier_stats = {}
        for r in rows:
            tier = r[6]
            if tier not in tier_stats:
                tier_stats[tier] = {'stakes': 0, 'pnl': 0, 'wins': 0, 'count': 0}
            tier_stats[tier]['stakes'] += r[0]
            tier_stats[tier]['pnl'] += r[2]
            tier_stats[tier]['wins'] += 1 if r[1] == 1 else 0
            tier_stats[tier]['count'] += 1
        
        # Save daily summary
        cursor.execute('''
            INSERT OR REPLACE INTO daily_summary 
            (date, starting_bankroll, ending_bankroll, total_staked, total_pnl, bets_count, wins, losses, roi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            report_date.isoformat(),
            self.current_bankroll - self.daily_pnl,
            self.current_bankroll,
            total_staked, total_pnl, len(rows), wins, losses, roi
        ))
        
        conn.commit()
        conn.close()
        
        report = {
            'date': report_date.isoformat(),
            'starting_bankroll': self.current_bankroll - self.daily_pnl,
            'ending_bankroll': self.current_bankroll,
            'total_staked': total_staked,
            'total_pnl': total_pnl,
            'roi': roi,
            'bets_count': len(rows),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(rows) if rows else 0,
            'tier_breakdown': tier_stats
        }
        
        # Save JSON report
        report_file = self.output_dir / f"daily_report_{report_date.isoformat()}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Daily report: {wins}W-{losses}L, PnL: {total_pnl:.2f}, ROI: {roi:.2%}")
        
        return report
    
    def get_performance_summary(self, days: int = 30) -> Dict:
        """Get performance summary for last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT date, total_pnl, total_staked, roi, bets_count, wins, losses
            FROM daily_summary
            WHERE date >= date('now', ?)
            ORDER BY date DESC
        ''', (f'-{days} days',))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {'message': 'No data'}
        
        total_pnl = sum(r[1] for r in rows)
        total_staked = sum(r[2] for r in rows)
        total_bets = sum(r[4] for r in rows)
        total_wins = sum(r[5] for r in rows)
        total_losses = sum(r[6] for r in rows)
        
        return {
            'period_days': days,
            'total_pnl': total_pnl,
            'total_staked': total_staked,
            'overall_roi': total_pnl / total_staked if total_staked > 0 else 0,
            'total_bets': total_bets,
            'wins': total_wins,
            'losses': total_losses,
            'win_rate': total_wins / total_bets if total_bets > 0 else 0,
            'avg_daily_pnl': total_pnl / len(rows),
            'profitable_days': sum(1 for r in rows if r[1] > 0),
            'losing_days': sum(1 for r in rows if r[1] < 0),
        }
    
    def reset_daily(self):
        """Reset daily counters"""
        self.daily_bets = []
        self.daily_pnl = 0.0
        self.bets_today = 0


def get_paper_trader(bankroll: float = 10000.0) -> PaperTradeLogger:
    return PaperTradeLogger(bankroll)