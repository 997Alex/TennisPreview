"""Core data models for TennisPreview"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import hashlib


class TournamentLevel(Enum):
    """Tennis tournament levels"""
    GS = "GS"           # Grand Slam
    MS1000 = "MS1000"   # Masters 1000
    ATP500 = "ATP500"
    ATP250 = "ATP250"
    WTA1000 = "WTA1000"
    WTA500 = "WTA500"
    WTA250 = "WTA250"
    WTA125 = "WTA125"
    CH175 = "CH175"     # Challenger 175
    CH125 = "CH125"
    CH100 = "CH100"
    CH75 = "CH75"
    CH50 = "CH50"
    CH25 = "CH25"
    ITF_W100 = "ITF_W100"
    ITF_W75 = "ITF_W75"
    ITF_W50 = "ITF_W50"
    ITF_W35 = "ITF_W35"
    ITF_W15 = "ITF_W15"
    ITF_M25 = "ITF_M25"
    ITF_M15 = "ITF_M15"
    OLYMPICS = "Olympics"
    QUALIES = "Qualies"
    JUNIOR = "Junior"
    EXHIBITION = "Exhibition"
    UNKNOWN = "UNKNOWN"


class Surface(Enum):
    HARD = "Hard"
    CLAY = "Clay"
    GRASS = "Grass"
    CARPET = "Carpet"
    HARD_INDOOR = "Hard (Indoor)"
    UNKNOWN = "Unknown"


class DataTier(Enum):
    TIER_1 = 1  # Full data available
    TIER_2 = 2  # Good data
    TIER_3 = 3  # Partial data
    TIER_4 = 4  # Market only


@dataclass
class Player:
    """Tennis player data"""
    id: str
    name: str
    ranking: int = 9999
    ranking_points: int = 0
    country: str = ""
    age: int = 0
    height_cm: int = 0
    handedness: str = "R"  # R, L, A
    backhand: str = "T"    # T=Two-handed, O=One-handed
    
    # Surface-specific stats
    surface_elo: Dict[Surface, float] = field(default_factory=dict)
    surface_win_pct: Dict[Surface, float] = field(default_factory=dict)
    surface_matches: Dict[Surface, int] = field(default_factory=dict)
    
    # Serve/Return ratings (0-100)
    serve_rating: Dict[Surface, float] = field(default_factory=dict)
    return_rating: Dict[Surface, float] = field(default_factory=dict)
    
    # Recent form
    matches_last_14d: int = 0
    sets_last_14d: int = 0
    wins_last_10: int = 0
    losses_last_10: int = 0
    
    # Fitness
    days_since_last_match: int = 0
    distance_travelled_km: float = 0.0
    timezone_diff_hours: float = 0.0
    
    def get_surface_elo(self, surface: Surface) -> float:
        return self.surface_elo.get(surface, 1500.0)
    
    def get_serve_rating(self, surface: Surface) -> float:
        return self.serve_rating.get(surface, 50.0)
    
    def get_return_rating(self, surface: Surface) -> float:
        return self.return_rating.get(surface, 50.0)


@dataclass
class TennisMatch:
    """Tennis match data"""
    id: str
    tournament: str
    tournament_level: TournamentLevel
    surface: Surface
    round: str
    best_of: int = 3
    
    player1: Player = None
    player2: Player = None
    
    scheduled_time: datetime = None
    status: str = "scheduled"  # scheduled, live, finished, cancelled
    
    # Scores (if finished)
    score: str = ""
    winner: str = ""
    sets: List[tuple] = field(default_factory=list)
    
    # Metadata
    court: str = ""
    umpire: str = ""
    
    def __post_init__(self):
        if self.id is None or self.id == "":
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique match ID"""
        p1 = self.player1.name if self.player1 else "TBD"
        p2 = self.player2.name if self.player2 else "TBD"
        date_str = self.scheduled_time.strftime("%Y%m%d") if self.scheduled_time else "nodate"
        raw = f"{p1}_{p2}_{self.tournament}_{date_str}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]
    
    @property
    def match_key(self) -> str:
        """Human readable match key"""
        p1 = self.player1.name if self.player1 else "TBD"
        p2 = self.player2.name if self.player2 else "TBD"
        return f"{p1} vs {p2}"


@dataclass
class OddsSnapshot:
    """Odds from a single bookmaker at a point in time"""
    bookmaker: str
    market: str  # h2h, spread, total
    odds_home: float
    odds_away: float
    timestamp: datetime
    source: str = "api"
    is_closing: bool = False
    volume: float = 0.0  # For exchange markets


@dataclass
class MergedOdds:
    """Merged odds from multiple sources"""
    match_id: str
    best_odds_home: float
    best_odds_away: float
    best_book_home: str
    best_book_away: str
    implied_prob_home: float
    implied_prob_away: float
    vig: float
    avg_odds_home: float
    avg_odds_away: float
    sources_count: int
    clv_estimate: Dict[str, float] = field(default_factory=dict)
    steam_detected: bool = False
    steam_side: str = ""
    wom_home: float = 0.0  # Weight of Money
    wom_away: float = 0.0
    last_update: datetime = None


@dataclass
class TennisFeatures:
    """Feature vector for tennis match prediction"""
    match_id: str
    
    # Player identifiers (for model lookup)
    player1_id: str = ""
    player2_id: str = ""
    
    # Player features
    p1_surface_elo: float = 1500.0
    p2_surface_elo: float = 1500.0
    p1_surface_win_pct: float = 0.5
    p2_surface_win_pct: float = 0.5
    p1_serve_rating: float = 50.0
    p2_serve_rating: float = 50.0
    p1_return_rating: float = 50.0
    p2_return_rating: float = 50.0
    
    # Form features
    p1_matches_14d: int = 0
    p2_matches_14d: int = 0
    p1_sets_14d: int = 0
    p2_sets_14d: int = 0
    p1_form_10: float = 0.5
    p2_form_10: float = 0.5
    
    # H2H
    h2h_total: int = 0
    h2h_p1_wins: int = 0
    h2h_surface: int = 0
    h2h_p1_wins_surface: int = 0
    
    # Context
    surface: Surface = Surface.HARD
    round: str = "R128"
    best_of: int = 3
    tournament_level: TournamentLevel = TournamentLevel.ATP250
    
    # Fitness
    p1_rest_days: int = 7
    p2_rest_days: int = 7
    p1_travel_km: float = 0.0
    p2_travel_km: float = 0.0
    p1_timezone_diff: float = 0.0
    p2_timezone_diff: float = 0.0
    
    # Market
    market_implied_p1: float = 0.5
    market_implied_p2: float = 0.5
    odds_movement_p1: float = 0.0
    odds_movement_p2: float = 0.0

    # Polymarket sentiment (crowd probability, 0.0 = assente)
    poly_implied_p1: float = 0.0
    poly_implied_p2: float = 0.0
    
    # Data quality
    data_tier: DataTier = DataTier.TIER_3
    feature_completeness: float = 0.5
    partial_profile: bool = False  # True if a side has no history (debutant)
    partial_side: str = ""  # 'player1' | 'player2' | 'both' | '' (which side is bare)
    
    # Best available odds (filled at merge time)
    best_odds_home: float = 0.0
    best_odds_away: float = 0.0
    
    # News adjustments
    p1_news_impact: float = 0.0
    p2_news_impact: float = 0.0
    
    def to_array(self) -> List[float]:
        """Convert to feature array for ML models"""
        return [
            self.p1_surface_elo, self.p2_surface_elo,
            self.p1_surface_win_pct, self.p2_surface_win_pct,
            self.p1_serve_rating, self.p2_serve_rating,
            self.p1_return_rating, self.p2_return_rating,
            self.p1_matches_14d, self.p2_matches_14d,
            self.p1_sets_14d, self.p2_sets_14d,
            self.p1_form_10, self.p2_form_10,
            self.h2h_total, self.h2h_p1_wins,
            self.h2h_surface, self.h2h_p1_wins_surface,
            self.best_of,
            self.p1_rest_days, self.p2_rest_days,
            self.p1_travel_km, self.p2_travel_km,
            self.p1_timezone_diff, self.p2_timezone_diff,
            self.market_implied_p1, self.market_implied_p2,
            self.odds_movement_p1, self.odds_movement_p2,
            self.poly_implied_p1, self.poly_implied_p2,
            self.feature_completeness,
            self.p1_news_impact, self.p2_news_impact,
        ]
    
    @property
    def feature_names(self) -> List[str]:
        return [
            'p1_surface_elo', 'p2_surface_elo',
            'p1_surface_win_pct', 'p2_surface_win_pct',
            'p1_serve_rating', 'p2_serve_rating',
            'p1_return_rating', 'p2_return_rating',
            'p1_matches_14d', 'p2_matches_14d',
            'p1_sets_14d', 'p2_sets_14d',
            'p1_form_10', 'p2_form_10',
            'h2h_total', 'h2h_p1_wins',
            'h2h_surface', 'h2h_p1_wins_surface',
            'best_of',
            'p1_rest_days', 'p2_rest_days',
            'p1_travel_km', 'p2_travel_km',
            'p1_timezone_diff', 'p2_timezone_diff',
            'market_implied_p1', 'market_implied_p2',
            'odds_movement_p1', 'odds_movement_p2',
            'poly_implied_p1', 'poly_implied_p2',
            'feature_completeness',
            'p1_news_impact', 'p2_news_impact',
        ]


@dataclass
class TennisPrediction:
    """Prediction result for a tennis match"""
    match_id: str
    p1_win_prob: float
    p2_win_prob: float
    model_probs: Dict[str, float]
    tier: DataTier
    confidence: float
    data_quality_flag: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Per-model breakdown
    bradley_terry_prob: float = 0.0
    glicko_prob: float = 0.0
    xgb_prob: float = 0.0
    elo_form_prob: float = 0.0
    market_prob: float = 0.0


@dataclass
class ValueBet:
    """Value bet recommendation"""
    match_id: str
    match_key: str
    tournament: str
    player1: str
    player2: str
    side: str  # "player1" or "player2"
    player_name: str
    odds: float
    model_prob: float
    implied_prob: float
    edge: float
    kelly_fraction: float
    stake_fraction: float
    tier: DataTier
    confidence: float
    data_quality: str
    clv_estimate: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def expected_value(self) -> float:
        return self.model_prob * self.odds - 1


@dataclass
class ApprovedBet:
    """Risk-approved bet with final stake (legacy: paper-money mode)."""
    value_bet: ValueBet
    stake: float
    bankroll: float
    risk_checks_passed: bool = True
    rejection_reason: str = ""


@dataclass
class BetDecision:
    """A value decision: who to back, at which real odds, and why.

    No money involved: the program outputs decisions, it never bets.
    """
    value_bet: ValueBet
    reason: str = ""
    rank: int = 0
    bookmaker: str = ""
    sisal: str = ""  # expected Sisal coverage ("si"/"no - ...")
    excluded_reason: str = ""  # set when filtered out (for transparency)