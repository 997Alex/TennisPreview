"""Daily pipeline orchestration for TennisPreview"""
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import asyncio
import json
from pathlib import Path

from src.utils.config import config
from src.utils.logging import get_logger, progress
from src.utils.models import (
    TennisMatch, TennisFeatures, TennisPrediction, MergedOdds,
    ValueBet, BetDecision, Player, Surface, TournamentLevel,
)
from src.data.ingest.sackmann import SackmannIngester
from src.data.ingest.tennis_data_co_uk import TennisDataCoUkIngester
from src.data.ingest.theoddsapi import TheOddsAPIClient
from src.data.ingest.api_football import APIFootballClient
from src.data.ingest.espn import (
    ESPNClient, normalize_name, name_tokens, surname_key,
)
from src.data.ingest.flashscore import FlashscoreScraper
from src.data.ingest.news import NewsIngester
from src.data.merge.odds_merger import OddsMerger
from src.sports.tennis.features.builder import TennisFeatureBuilder
from src.sports.tennis.models.ensemble import TennisEnsemble
from src.betting.value_detector import (
    TennisValueDetector, explain_value_bet, sisal_coverage,
)

logger = get_logger("daily_pipeline")


class PipelineResult:
    """Container for pipeline results (used for console display)"""

    def __init__(self):
        self.matches: List[TennisMatch] = []
        self.predictions: Dict[str, TennisPrediction] = {}
        self.odds: Dict[str, MergedOdds] = {}
        self.features: Dict[str, TennisFeatures] = {}
        self.value_bets: List[ValueBet] = []
        self.decisions: List[BetDecision] = []
        self.excluded: List[BetDecision] = []
        self.poly: Dict[str, tuple] = {}
        self.demo_mode = False
        self.demo_odds = False  # solo sandbox --demo
        self.decision_summary: Dict = {}
        self.news: Dict[str, Dict] = {}  # impatto news per giocatore
        self.sisal_flag: Dict[str, tuple] = {}  # match_id -> ("si"|"no", nota)
    
    @property
    def analyzed_events(self) -> int:
        return len(self.predictions)


class TennisDailyPipeline:
    """Main daily pipeline for tennis match analysis"""
    
    def __init__(self, demo_mode: bool = False):
        self.config = config.get('tennis')
        self.app_config = config.get('app')
        self.demo_mode = demo_mode

        # Initialize components (no money: solo decisioni, mai scommesse)
        self.sackmann = SackmannIngester()
        self.tennis_data = TennisDataCoUkIngester()
        self.theoddsapi = TheOddsAPIClient()
        self.api_football = APIFootballClient()
        self.espn = ESPNClient()
        self.betexplorer = None  # lazy (requests solo se serve)
        self.polymarket = None  # lazy (requests solo se serve)
        self.flashscore = FlashscoreScraper()
        self.news = NewsIngester()

        self.odds_merger = OddsMerger()
        self.feature_builder = TennisFeatureBuilder()
        self.ensemble = TennisEnsemble()
        self.value_detector = TennisValueDetector()
        
        # Data caches
        self.player_db = {}
        self._name_index = {}
        self.historical_matches = []
        self.h2h_cache = {}
        self._h2h_index = {}
        self.training_features = []
        self.training_labels = []

        self._models_fitted = False
    
    async def run(self, target_date: date = None) -> PipelineResult:
        """Run the complete daily pipeline"""
        target_date = target_date or date.today()
        result = PipelineResult()
        result.demo_mode = self.demo_mode

        logger.info(f"Starting daily pipeline for {target_date}")
        print()

        try:
            # 1. Load/update historical data
            await self._load_historical_data()

            # 2. Get today's matches
            matches = await self._get_matches_for_date(target_date)

            if not matches:
                logger.warning("No matches found online - falling back to demo mode")
                matches = self._build_demo_matches(target_date)
                result.demo_mode = True

            result.matches = matches
            logger.info(f"Found {len(matches)} matches for {target_date}")

            # 2b. Filtro Sisal si/no per OGNI match (come prima):
            #     si se nel palinsesto Sisal reale, altrimenti stima per tier
            result.sisal_flag = {
                m.id: self._sisal_presence(m) for m in matches
            }
            n_si = sum(1 for f in result.sisal_flag.values() if f[0] == "si")
            logger.info(f"Sisal: {n_si}/{len(matches)} match presenti")

            # 3. Get odds: demo ONLY in explicit --demo sandbox.
            #    On live runs only real bookmaker odds are used; matches
            #    without real odds stay n/a and are excluded from bets.
            use_demo_odds = result.demo_mode
            result.demo_odds = bool(use_demo_odds)
            odds_dict = await self._get_odds_for_matches(matches, demo=use_demo_odds)
            result.odds = odds_dict

            # 3b. Polymarket sentiment per match (peso nel pronostico)
            poly_dict = self._get_poly_sentiment(matches, demo=result.demo_mode)
            result.poly = poly_dict

            # 4. Get news impact
            news_impact = self._get_news_impact(matches)
            result.news = news_impact

            # 5. Build features
            features_dict = self._build_features(matches, odds_dict, news_impact,
                                                 poly_dict)
            result.features = features_dict
            
            # 6. Ensure models are fitted
            if not self._models_fitted:
                self._fit_models()
            
            # 7. Generate predictions
            predictions = self._generate_predictions(features_dict)
            result.predictions = predictions
            
            # 8. Find value bets
            value_bets = self._find_value_bets(predictions, odds_dict, matches)
            result.value_bets = value_bets

            # 9. Decisioni finali (filtri anti-duplicazione, niente soldi)
            decisions, excluded = self._select_decisions(
                value_bets, features_dict, odds_dict)
            result.decisions = decisions
            result.excluded = excluded

            # Sintesi per la console
            by_tier: Dict[str, int] = {}
            for d in decisions:
                t = d.value_bet.tier.value if hasattr(d.value_bet.tier, 'value') else d.value_bet.tier
                by_tier[f"Tier {t}"] = by_tier.get(f"Tier {t}", 0) + 1
            result.decision_summary = {
                'decisions': len(decisions),
                'value_found': len(value_bets),
                'excluded': len(excluded),
                'by_tier': by_tier,
                'sisal_yes': sum(1 for d in decisions if d.sisal == "si"),
            }

            # 10. Generate report
            self._generate_report(target_date, result)

            logger.info(f"Pipeline complete. {len(decisions)} decisioni finali "
                        f"({len(excluded)} scartate).")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    async def _load_historical_data(self, recent_years: int = 6, force: bool = False):
        """Load historical data from local caches / sources.

        Only the most recent seasons are loaded: using the full archive
        (1968-today, ~1.8M rows incl. futures/qualies) makes profile
        building + model fitting take >10 minutes. The last `recent_years`
        seasons are more than enough for Elo/form features and ML training.
        Results are cached on the instance so multi-day runs (--days 7)
        don't reload + refit every day.
        """
        if self.historical_matches and self.player_db and not force:
            logger.info("Historical data already loaded - reusing cache")
            return
        # Reset derived caches when (re)loading
        self.training_features = []
        self.training_labels = []
        self.h2h_cache = {}
        self._h2h_index = {}
        self._name_index = {}
        logger.info("Loading historical data...")

        progress.start(3, "Historical data")
        progress.set_desc("Loading historical data: Sackmann repos")

        # Update Sackmann repos (best-effort, offline-safe)
        self.sackmann.update_repos()
        progress.update()

        from datetime import date as _date
        current_year = _date.today().year
        years = list(range(current_year - recent_years + 1, current_year + 1))

        progress.set_desc(f"Loading historical data: match files {years[0]}-{years[-1]}")
        atp_matches = self.sackmann.load_matches('atp', years=years)
        wta_matches = self.sackmann.load_matches('wta', years=years)
        progress.update()

        self.historical_matches = []
        for df in [atp_matches, wta_matches]:
            if df is not None and not df.empty:
                self.historical_matches.extend(df.to_dict('records'))

        progress.set_desc("Loading historical data: player profiles")
        atp_rankings = self.sackmann.load_rankings('atp')
        wta_rankings = self.sackmann.load_rankings('wta')

        self.player_db = self.sackmann.build_player_profiles(atp_matches, atp_rankings)
        wta_players_db = self.sackmann.build_player_profiles(wta_matches, wta_rankings)
        self.player_db.update(wta_players_db)
        self._build_name_index()
        progress.update()
        progress.close()

        logger.info(f"Loaded {len(self.historical_matches):,} historical matches")
        logger.info(f"Built profiles for {len(self.player_db):,} players")

        # Pre-build H2H index once (avoids O(N*M) scans later)
        self._build_h2h_index()

        # Fresh results (EVERY startup): ultimo anno ESPN rigiocato su
        # profili/H2H/training cosi' forma e storico sono sempre aggiornati.
        # Le date gia' scaricate sono cachate (data/recent/espn_fetched.json):
        # il primo avvio impiega ~15-20 min, i successivi solo i giorni nuovi.
        try:
            recent = self._backfill_history(days=365)
            if recent:
                self._apply_recent_results(recent)
                # new bare names must resolve in live matching too
                self._build_name_index()
        except Exception as e:
            logger.warning(f"Recent-results update skipped: {e}")

        # Build training features/labels for ML models
        self._build_training_set()
    
    def _build_training_set(self, max_samples: int = 15000):
        """Build training features from historical matches for ML models"""
        logger.info("Building training set from historical data...")
        
        if not self.historical_matches:
            logger.warning("No historical matches available for training")
            return
        
        # Sort chronologically (oldest first)
        hist = sorted(
            self.historical_matches,
            key=lambda m: str(m.get('tourney_date', '19000101'))
        )
        
        if len(hist) > max_samples:
            hist = hist[-max_samples:]

        import random
        rng = random.Random(42)

        progress.start(len(hist), "Building training set")
        for match in hist:
            try:
                winner_id = match.get('winner_id')
                loser_id = match.get('loser_id')
                if not winner_id or not loser_id:
                    progress.update()
                    continue

                winner = self.player_db.get(str(winner_id))
                loser = self.player_db.get(str(loser_id))
                if not winner or not loser:
                    progress.update()
                    continue

                surface = self.sackmann.parse_surface(match.get('surface', 'Hard'))
                level = self.sackmann.parse_tournament_level(
                    match.get('tourney_level', ''), match.get('tour', 'atp')
                )

                raw_date = match.get('tourney_date', '20200101')
                if hasattr(raw_date, 'strftime'):
                    match_dt = raw_date.to_pydatetime() if hasattr(raw_date, 'to_pydatetime') else raw_date
                else:
                    try:
                        match_dt = datetime.strptime(str(int(float(str(raw_date)))) if str(raw_date).replace('.', '', 1).isdigit() else str(raw_date), '%Y%m%d')
                    except (ValueError, TypeError):
                        match_dt = datetime(2020, 1, 1)

                # Randomize orientation so labels are balanced:
                # ~50% winner as player1 (label 1), ~50% as player2 (label 0).
                # Without this the training set has a single class and both
                # the sklearn models and the isotonic calibrators collapse
                # to predicting ~1.0 for every match.
                winner_is_p1 = rng.random() < 0.5
                if winner_is_p1:
                    p1, p2, label = winner, loser, 1
                else:
                    p1, p2, label = loser, winner, 0

                m = TennisMatch(
                    id=f"hist_{str(raw_date)}_{winner_id}_{loser_id}_{int(winner_is_p1)}",
                    tournament=match.get('tourney_name', 'Historical'),
                    tournament_level=level,
                    surface=surface,
                    round=match.get('round', 'R64'),
                    player1=p1,
                    player2=p2,
                    scheduled_time=match_dt
                )

                feat = self.feature_builder.build_features(
                    match=m,
                    player_db=self.player_db,
                    merged_odds=None,
                    news_impact=None,
                    h2h_data=None
                )
                self.training_features.append(feat)
                self.training_labels.append(label)
                progress.update()

            except Exception:
                progress.update()
                continue

        progress.close()
        logger.info(f"Training set built: {len(self.training_features)} samples")
    
    def _has_api_key(self, client, attr: str = 'api_key') -> bool:
        try:
            key = getattr(client, attr, '')
            return bool(key and key != 'your_key_here' and len(str(key)) > 8)
        except Exception:
            return False

    def _flashscore_to_match(self, fsm) -> Optional[TennisMatch]:
        """Convert FlashscoreMatch (or dict) to TennisMatch."""
        try:
            from src.utils.models import Player as _Player
            if isinstance(fsm, TennisMatch):
                return fsm
            mid = getattr(fsm, 'id', None) or (fsm.get('id') if isinstance(fsm, dict) else None)
            if not mid:
                return None
            if isinstance(fsm, dict):
                p1n = fsm.get('player1', 'Unknown')
                p2n = fsm.get('player2', 'Unknown')
                tourn = fsm.get('tournament', 'Flashscore Event')
                surface = fsm.get('surface', Surface.HARD)
                rnd = fsm.get('round', 'TBD')
                st = fsm.get('start_time') or datetime.now()
            else:
                p1n = getattr(fsm, 'player1', 'Unknown')
                p2n = getattr(fsm, 'player2', 'Unknown')
                tourn = getattr(fsm, 'tournament', 'Flashscore Event')
                surface = getattr(fsm, 'surface', Surface.HARD)
                rnd = getattr(fsm, 'round', 'TBD')
                st = getattr(fsm, 'start_time', None) or datetime.now()
            if isinstance(surface, str):
                try:
                    surface = Surface(surface)
                except ValueError:
                    surface = Surface.HARD
            # stats dict may carry surface info; keep HARD default
            p1 = _Player(id=str(p1n), name=str(p1n))
            p2 = _Player(id=str(p2n), name=str(p2n))
            if hasattr(st, 'replace') and getattr(st, 'tzinfo', None) is not None:
                try:
                    st = st.replace(tzinfo=None)
                except Exception:
                    pass
            return TennisMatch(
                id=f"fs_{mid}",
                tournament=str(tourn),
                tournament_level=TournamentLevel.UNKNOWN,
                surface=surface if isinstance(surface, Surface) else Surface.HARD,
                round=str(rnd),
                player1=p1,
                player2=p2,
                scheduled_time=st,
            )
        except Exception:
            return None

    async def _get_matches_for_date(self, target_date: date) -> List[TennisMatch]:
        """Get matches for a specific date from multiple sources"""
        if self.demo_mode:
            return self._build_demo_matches(target_date)

        all_matches: Dict[str, TennisMatch] = {}

        # Try ESPN first: free, no key, real current-week matches (attualita)
        try:
            logger.info("Scanning upcoming matches via ESPN...")
            for em in self.espn.get_matches_for_date(target_date):
                match = self._espn_to_match(em)
                if match and match.id not in all_matches:
                    all_matches[match.id] = match
            logger.info(f"ESPN: {len(all_matches)} match live/schedulati")
        except Exception as e:
            logger.warning(f"ESPN matches failed: {e}")

        # Sisal.it: palinsesto del giorno (fonte primaria richiesta)
        self._sisal_list = []
        self._sisal_pairs = set()
        try:
            from src.data.ingest.sisal import SisalScraper
            logger.info("Scanning palinsesto Sisal del giorno...")
            self._sisal_list = await SisalScraper().get_matches_for_date(target_date)
            for sm in self._sisal_list:
                try:
                    a = normalize_name(sm.player1)
                    b = normalize_name(sm.player2)
                    if a and b:
                        self._sisal_pairs.add((a, b))
                        self._sisal_pairs.add((b, a))
                except Exception:
                    continue
            have_pairs = set()
            for m in all_matches.values():
                try:
                    if m.player1 and m.player2:
                        a = normalize_name(m.player1.name)
                        b = normalize_name(m.player2.name)
                        have_pairs.add((a, b))
                        have_pairs.add((b, a))
                except Exception:
                    continue
            for sm in self._sisal_list:
                match = self._sisal_to_match(sm, target_date)
                if not match or match.id in all_matches:
                    continue
                try:
                    pair = (normalize_name(match.player1.name),
                            normalize_name(match.player2.name))
                    if pair in have_pairs:
                        continue  # gia' presente da ESPN: le quote si uniscono dopo
                    have_pairs.add(pair)
                    have_pairs.add((pair[1], pair[0]))
                except Exception:
                    pass
                all_matches[match.id] = match
            logger.info(f"Sisal: {len(all_matches)} match totali finora")
        except Exception as e:
            logger.warning(f"Sisal matches failed: {e}")

        # Try TheOddsAPI first (has both fixtures and odds) - skip if no key
        if self._has_api_key(self.theoddsapi):
            try:
                logger.info("Scanning upcoming matches via TheOddsAPI...")
                odds_matches = self.theoddsapi.get_tennis_odds()
                for m in odds_matches:
                    if m.id not in all_matches:
                        match = self._oddsapi_to_match(m)
                        if match:
                            all_matches[match.id] = match
            except Exception as e:
                logger.warning(f"TheOddsAPI matches failed: {e}")
        else:
            logger.info("TheOddsAPI key missing - skipping live odds scan")

        # Try API-Football - skip if no key
        if self._has_api_key(self.api_football):
            try:
                logger.info("Scanning upcoming matches via API-Football...")
                fixtures = self.api_football.get_fixtures(target_date)
                for fix in fixtures:
                    match = self._fixture_to_match(fix)
                    if match and match.id not in all_matches:
                        all_matches[match.id] = match
            except Exception as e:
                logger.warning(f"API-Football fixtures failed: {e}")
        else:
            logger.info("API-Football key missing - skipping fixtures scan")

        # Try Flashscore (no key needed, but parser is best-effort)
        try:
            logger.info("Scanning upcoming matches via Flashscore...")
            fs_matches = await self.flashscore.get_today_matches()
            for m in fs_matches:
                conv = self._flashscore_to_match(m)
                if conv and conv.id not in all_matches:
                    all_matches[conv.id] = conv
        except Exception as e:
            logger.warning(f"Flashscore matches failed: {e}")
        finally:
            try:
                await self.flashscore.close()
            except Exception:
                pass

        return list(all_matches.values())
    
    def _espn_to_match(self, em) -> Optional[TennisMatch]:
        """Convert ESPN live match to TennisMatch (players resolved later)."""
        try:
            p1 = Player(id=str(em.player1), name=str(em.player1))
            p2 = Player(id=str(em.player2), name=str(em.player2))
            try:
                level = TournamentLevel(em.level_hint)
            except ValueError:
                level = self._map_league_to_level(em.tournament)
            try:
                surface = Surface(em.surface_hint)
            except ValueError:
                surface = Surface.HARD
            status = {"pre": "scheduled", "in": "live", "post": "finished"}.get(
                getattr(em, "status", "pre"), "scheduled")
            return TennisMatch(
                id=str(em.id),
                tournament=str(em.tournament or "Unknown"),
                tournament_level=level,
                surface=surface,
                round=str(em.round or "TBD"),
                best_of=int(getattr(em, "best_of", 3) or 3),
                player1=p1,
                player2=p2,
                scheduled_time=getattr(em, "scheduled_time", None) or datetime.now(),
                status=status,
            )
        except Exception:
            return None

    def _sisal_presence(self, match: TennisMatch) -> tuple:
        """Filtro si/no presenza su piattaforma Sisal (come prima).

        'si' se il match e' nel palinsesto Sisal reale (scraping/cache),
        altrimenti stima per tier (solo tour maggiore -> 'si').
        """
        try:
            if getattr(match, "sisal_odds", None):
                return ("si", "nel palinsesto Sisal")
            if match.player1 and match.player2:
                key = (normalize_name(match.player1.name),
                       normalize_name(match.player2.name))
                if key in getattr(self, "_sisal_pairs", set()):
                    return ("si", "nel palinsesto Sisal")
            tier = self.feature_builder.determine_tier(match)
            return sisal_coverage(tier)
        except Exception:
            return ("no", "verifica su Sisal.it")

    def _sisal_presence_bet(self, bet: ValueBet, odds) -> tuple:
        """Filtro si/no Sisal per una decisione (palinsesto reale o stima tier)."""
        try:
            if odds and ("sisal" in str(getattr(odds, "best_book_home", "")) or
                         "sisal" in str(getattr(odds, "best_book_away", ""))):
                return ("si", "nel palinsesto Sisal")
            if bet.player1 and bet.player2:
                key = (normalize_name(bet.player1), normalize_name(bet.player2))
                if key in getattr(self, "_sisal_pairs", set()):
                    return ("si", "nel palinsesto Sisal")
            return sisal_coverage(bet.tier)
        except Exception:
            return ("no", "verifica su Sisal.it")

    def _sisal_to_match(self, sm, target_date: date) -> Optional[TennisMatch]:
        """Convert Sisal palinsesto row to TennisMatch (quote in extra)."""
        try:
            if not sm.player1 or not sm.player2:
                return None
            p1 = Player(id=f"sisal_{sm.player1}", name=str(sm.player1))
            p2 = Player(id=f"sisal_{sm.player2}", name=str(sm.player2))
            level = self._map_league_to_level(sm.tournament)
            st = sm.start_time or datetime.combine(target_date, datetime.min.time())
            if hasattr(st, "replace") and getattr(st, "tzinfo", None) is not None:
                try:
                    st = st.replace(tzinfo=None)
                except Exception:
                    pass
            m = TennisMatch(
                id=str(sm.id),
                tournament=str(sm.tournament or "Sisal Tennis"),
                tournament_level=level,
                surface=Surface.HARD,
                round="TBD",
                player1=p1,
                player2=p2,
                scheduled_time=st,
            )
            # Sisal odds attached for the merger (non-invasive: dict attr)
            m.sisal_odds = (sm.odds1, sm.odds2)  # type: ignore[attr-defined]
            return m
        except Exception:
            return None

    def _fixture_to_match(self, fixture) -> Optional[TennisMatch]:
        """Convert API-Football fixture to TennisMatch"""
        try:
            teams = fixture.teams
            home = teams.get('home', {})
            away = teams.get('away', {})
            
            home_name = home.get('name', '') or 'Unknown'
            away_name = away.get('name', '') or 'Unknown'
            
            if not home_name or not away_name:
                return None
            
            p1 = Player(id=str(home.get('id', home_name)), name=home_name)
            p2 = Player(id=str(away.get('id', away_name)), name=away_name)
            
            # Map league to tournament level
            league_name = fixture.league.get('name', '') if fixture.league else ''
            level = self._map_league_to_level(league_name)
            
            return TennisMatch(
                id=f"af_{fixture.id}",
                tournament=league_name or "Unknown",
                tournament_level=level,
                surface=Surface.HARD,
                round=fixture.status.get('short', 'NS'),
                player1=p1,
                player2=p2,
                scheduled_time=fixture.date
            )
        except Exception:
            return None
    
    def _map_league_to_level(self, league_name: str) -> TournamentLevel:
        """Map league name to tournament level"""
        name_lower = (league_name or '').lower()
        mapping = {
            'wimbledon': TournamentLevel.GS,
            'us open': TournamentLevel.GS,
            'australian open': TournamentLevel.GS,
            'roland': TournamentLevel.GS,
            'french open': TournamentLevel.GS,
            'masters': TournamentLevel.MS1000,
            'atp 500': TournamentLevel.ATP500,
            'atp 250': TournamentLevel.ATP250,
            'wta 1000': TournamentLevel.WTA1000,
            'wta 500': TournamentLevel.WTA500,
            'wta 250': TournamentLevel.WTA250,
            'challenger': TournamentLevel.CH100,
        }
        for key, level in mapping.items():
            if key in name_lower:
                return level
        return TournamentLevel.UNKNOWN
    
    def _oddsapi_to_match(self, odds_match) -> Optional[TennisMatch]:
        """Convert TheOddsAPI match to TennisMatch"""
        try:
            p1 = Player(id=odds_match.home_team, name=odds_match.home_team)
            p2 = Player(id=odds_match.away_team, name=odds_match.away_team)
            
            return TennisMatch(
                id=odds_match.id,
                tournament="Upcoming Tournament",
                tournament_level=TournamentLevel.UNKNOWN,
                surface=Surface.HARD,
                round="TBD",
                player1=p1,
                player2=p2,
                scheduled_time=odds_match.commence_time.replace(tzinfo=None)
            )
        except Exception:
            return None
    
    def _build_demo_matches(self, target_date: date) -> List[TennisMatch]:
        """Build demo matches from historical data (offline mode)"""
        logger.info("Building demo matches from historical data...")
        matches = []
        if not self.historical_matches:
            # Nessun DB storico (Sackmann non scaricato): 6 match sintetici
            # con nomi reali cosi' --demo mostra comunque la pipeline.
            logger.info("No historical DB: uso match sintetici dimostrativi")
            synth = [
                ("Jannik Sinner", "Carlos Alcaraz", "ATP500", Surface.HARD),
                ("Novak Djokovic", "Daniil Medvedev", "GS", Surface.HARD),
                ("Iga Swiatek", "Aryna Sabalenka", "WTA1000", Surface.CLAY),
                ("Alexander Zverev", "Stefanos Tsitsipas", "MS1000", Surface.CLAY),
                ("Coco Gauff", "Jessica Pegula", "WTA500", Surface.HARD),
                ("Matteo Berrettini", "Lorenzo Musetti", "ATP250", Surface.GRASS),
            ]
            lvl_map = {"GS": TournamentLevel.GS, "MS1000": TournamentLevel.MS1000,
                       "ATP500": TournamentLevel.ATP500, "ATP250": TournamentLevel.ATP250,
                       "WTA1000": TournamentLevel.WTA1000, "WTA500": TournamentLevel.WTA500}
            for i, (a, b, lvl, surf) in enumerate(synth):
                matches.append(TennisMatch(
                    id=f"demo_{target_date.isoformat()}_{i}",
                    tournament="Demo Event", tournament_level=lvl_map[lvl],
                    surface=surf, round="R32",
                    player1=Player(id=f"demo_{a}", name=a),
                    player2=Player(id=f"demo_{b}", name=b),
                    scheduled_time=datetime.combine(
                        target_date, datetime.min.time()) + timedelta(hours=11 + i),
                ))
            logger.info(f"Built {len(matches)} synthetic demo matches")
            return matches
        
        # Take recent historical matchups as demo events
        hist_sorted = sorted(
            self.historical_matches,
            key=lambda m: str(m.get('tourney_date', '19000101')),
            reverse=True
        )
        
        seen_pairs = set()
        for match in hist_sorted[:200]:
            try:
                winner_id = match.get('winner_id')
                loser_id = match.get('loser_id')
                winner_name = match.get('winner_name', '')
                loser_name = match.get('loser_name', '')
                
                if not winner_id or not loser_id or not winner_name or not loser_name:
                    continue
                
                pair_key = tuple(sorted([winner_name, loser_name]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                p1 = Player(id=str(winner_id), name=winner_name)
                p2 = Player(id=str(loser_id), name=loser_name)
                
                surface = self.sackmann.parse_surface(match.get('surface', 'Hard'))
                level = self.sackmann.parse_tournament_level(
                    match.get('tourney_level', ''), match.get('tour', 'atp')
                )
                
                matches.append(TennisMatch(
                    id=f"demo_{target_date.isoformat()}_{len(matches)}",
                    tournament=match.get('tourney_name', 'Demo Event'),
                    tournament_level=level,
                    surface=surface,
                    round=match.get('round', 'R64'),
                    player1=p1,
                    player2=p2,
                    scheduled_time=datetime.combine(
                        target_date, datetime.min.time()
                    ) + timedelta(hours=10 + len(matches))
                ))
                
                if len(matches) >= 10:
                    break
            except Exception:
                continue
        
        logger.info(f"Built {len(matches)} demo matches")
        return matches
    
    async def _get_odds_for_matches(self, matches: List[TennisMatch],
                                    demo: bool = False) -> Dict[str, MergedOdds]:
        """Collect REAL odds from all sources (never invented on live).

        demo=True only in explicit --demo sandbox mode. On live runs every
        match without real odds stays n/a and is excluded from value bets.
        """
        if demo:
            return self._generate_demo_odds(matches)

        match_ids = [m.id for m in matches]
        all_odds: Dict[str, List] = {}

        progress.start(4, "Collecting odds")

        # TheOddsAPI (key gratuita): quote reali unite per nome normalizzato
        progress.set_desc("Collecting odds: TheOddsAPI")
        if self._has_api_key(self.theoddsapi):
            try:
                toa_map = self.theoddsapi.get_odds_by_names()
                n_toa = 0
                for match in matches:
                    if not match.player1 or not match.player2:
                        continue
                    try:
                        key = (normalize_name(match.player1.name),
                               normalize_name(match.player2.name))
                        hit = toa_map.get(key)
                        if hit:
                            all_odds.setdefault(match.id, []).extend(hit)
                            n_toa += 1
                    except Exception:
                        continue
                logger.info(f"TheOddsAPI: quote reali per {n_toa} match")
            except Exception as e:
                logger.warning(f"TheOddsAPI odds failed: {e}")
        else:
            logger.info("TheOddsAPI key missing - skipping odds fetch")
        progress.update()

        # Sisal (primary): quote reali dal palinsesto gia' scansionato
        # + join per nome normalizzato (match ESPN <-> quote Sisal cache)
        progress.set_desc("Collecting odds: Sisal")
        try:
            from src.utils.models import OddsSnapshot as _Snap
            from datetime import datetime as _dt
            n_sisal = 0
            for match in matches:
                pair = getattr(match, "sisal_odds", None)
                if pair and pair[0] > 1.0 and pair[1] > 1.0:
                    all_odds.setdefault(match.id, []).append(_Snap(
                        bookmaker="sisal", market="h2h",
                        odds_home=float(pair[0]), odds_away=float(pair[1]),
                        timestamp=_dt.now(), source="sisal",
                    ))
                    n_sisal += 1
            if getattr(self, "_sisal_list", None):
                sis_map = {}
                for sm in self._sisal_list:
                    try:
                        a = normalize_name(sm.player1)
                        b = normalize_name(sm.player2)
                        if a and b and sm.odds1 > 1.0 and sm.odds2 > 1.0:
                            sis_map[(a, b)] = (sm.odds1, sm.odds2)
                            sis_map[(b, a)] = (sm.odds2, sm.odds1)
                    except Exception:
                        continue
                for match in matches:
                    if match.id in all_odds or not match.player1 or not match.player2:
                        continue
                    try:
                        key = (normalize_name(match.player1.name),
                               normalize_name(match.player2.name))
                        hit = sis_map.get(key)
                        if hit:
                            all_odds.setdefault(match.id, []).append(_Snap(
                                bookmaker="sisal", market="h2h",
                                odds_home=float(hit[0]), odds_away=float(hit[1]),
                                timestamp=_dt.now(), source="sisal",
                            ))
                            n_sisal += 1
                    except Exception:
                        continue
            logger.info(f"Sisal: quote reali per {n_sisal} match")
        except Exception as e:
            logger.warning(f"Sisal odds failed: {e}")
        progress.update()

        # Betexplorer (free): best-comparison odds for live tournaments
        progress.set_desc("Collecting odds: Betexplorer")
        try:
            from src.data.ingest.betexplorer import BetexplorerClient
            if self.betexplorer is None:
                self.betexplorer = BetexplorerClient()
            be_odds = self.betexplorer.get_odds_for_matches(matches)
            for mid, odds_list in be_odds.items():
                all_odds.setdefault(mid, []).extend(odds_list)
            logger.info(f"Betexplorer: quote vere per {len(be_odds)} match")
        except Exception as e:
            logger.warning(f"Betexplorer odds failed: {e}")
        progress.update()

        # Polymarket (free): real exchange prices for Tier-1 gaps
        progress.set_desc("Collecting odds: Polymarket")
        try:
            from src.data.ingest.polymarket import PolymarketClient
            from src.utils.models import TournamentLevel as _TL
            big = [m for m in matches
                   if m.id not in all_odds
                   and m.tournament_level in (_TL.GS, _TL.MS1000, _TL.WTA1000)]
            if big:
                if self.polymarket is None:
                    self.polymarket = PolymarketClient()
                pm_odds = self.polymarket.get_odds_for_matches(big)
                for mid, odds_list in pm_odds.items():
                    all_odds.setdefault(mid, []).extend(odds_list)
                logger.info(f"Polymarket: quote vere per {len(pm_odds)} match")
        except Exception as e:
            logger.warning(f"Polymarket odds failed: {e}")
        progress.update()

        # API-Football (key)
        progress.set_desc("Collecting odds: API-Football")
        if self._has_api_key(self.api_football):
            try:
                for match in matches:
                    af_odds = self.api_football.get_odds(match.id)
                    if af_odds:
                        all_odds.setdefault(match.id, []).extend(af_odds)
            except Exception as e:
                logger.warning(f"API-Football odds failed: {e}")
        else:
            logger.info("API-Football key missing - skipping odds fetch")
        progress.update()
        progress.close()

        # Merge odds
        merged = self.odds_merger.merge(all_odds)
        n_missing = len(matches) - len(merged)
        logger.info(f"Merged REAL odds for {len(merged)} matches "
                    f"({n_missing} senza quota -> n/a, esclusi dai bet)")
        return merged
    
    def _generate_demo_odds(self, matches: List[TennisMatch]) -> Dict[str, MergedOdds]:
        """Generate realistic demo odds for offline testing"""
        import numpy as np
        from src.utils.models import OddsSnapshot
        
        demo_odds: Dict[str, MergedOdds] = {}
        
        for match in matches:
            # Prefer full profiles from player_db for realistic odds
            # (by id or normalized name); new players fall back to 50/50.
            p1 = self._resolve_player(match.player1) if match.player1 else None
            p2 = self._resolve_player(match.player2) if match.player2 else None

            # Derive pseudo-Elo from surface Elo when available, else win%.
            if p1 is not None and match.surface in getattr(p1, 'surface_elo', {}):
                elo1 = p1.surface_elo[match.surface]
            else:
                wp1 = p1.surface_win_pct.get(match.surface, 0.5) if p1 else 0.5
                elo1 = 1500 + (wp1 - 0.5) * 600
            if p2 is not None and match.surface in getattr(p2, 'surface_elo', {}):
                elo2 = p2.surface_elo[match.surface]
            else:
                wp2 = p2.surface_win_pct.get(match.surface, 0.5) if p2 else 0.5
                elo2 = 1500 + (wp2 - 0.5) * 600
            p1_prob = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
            p1_prob = float(np.clip(p1_prob, 0.08, 0.92))
            
            # Add small random noise to simulate market inefficiency
            noise = np.random.normal(0, 0.015)
            market_p1 = float(np.clip(p1_prob + noise, 0.05, 0.95))
            
            # Generate odds with vig (~5%)
            vig = 0.05
            raw_odds_home = 1 / (market_p1 * (1 + vig))
            raw_odds_away = 1 / ((1 - market_p1) * (1 + vig))
            
            # Create fake snapshots from "demo" bookmakers
            snapshots = []
            now = datetime.now()
            for bm, adj in [("pinnacle", 0.98), ("bet365", 1.0), ("williamhill", 1.02)]:
                snapshots.append(OddsSnapshot(
                    bookmaker=bm,
                    market='h2h',
                    odds_home=max(1.01, raw_odds_home * adj),
                    odds_away=max(1.01, raw_odds_away * adj),
                    timestamp=now,
                    source='demo'
                ))
            
            merged = self.odds_merger._merge_match(match.id, snapshots)
            if merged:
                demo_odds[match.id] = merged
        
        return demo_odds
    
    def _get_news_impact(self, matches: List[TennisMatch]) -> Dict[str, Dict]:
        """Get news impact for all players in matches"""
        all_players = set()
        for m in matches:
            if m.player1:
                all_players.add(m.player1.name)
            if m.player2:
                all_players.add(m.player2.name)
        
        news_impact: Dict[str, Dict] = {}
        articles = []
        
        if not self.demo_mode:
            try:
                logger.info("Fetching news for players...")
                articles = self.news.fetch_all_feeds(hours_back=72)
                logger.info(f"  {len(articles)} articles fetched")
            except Exception as e:
                logger.warning(f"News fetch failed: {e}")
        
        for player in all_players:
            try:
                impact = self.news.get_player_news_impact(
                    player, hours_back=72, articles=articles
                )
            except Exception:
                impact = {'impact': 0.0, 'confidence': 0.0, 'article_count': 0}
            news_impact[player] = impact
        
        return news_impact
    
    def _build_name_index(self):
        """Index Sackmann profiles by normalized name for live matching.

        Live sources (ESPN) provide names, not Sackmann ids: this lets a
        current match with history resolve to its full profile (Elo, form,
        H2H). Unknown names simply miss and take the new-player path.
        """
        self._name_index = {}
        self._fuzzy_index = []  # (surname_key, first_name, normkey, player)
        for pid, player in self.player_db.items():
            try:
                key = normalize_name(player.name)
                if key and key not in self._name_index:
                    self._name_index[key] = player
                toks = name_tokens(player.name)
                if toks:
                    self._fuzzy_index.append(
                        ("".join(toks[1:]) if len(toks) > 1 else "".join(toks),
                         toks[0], key, player))
            except Exception:
                continue
        logger.info(f"Name index built: {len(self._name_index)} names")

    def _fuzzy_player(self, name: str):
        """Transliteration variants ('Sobolieva' vs 'Soboleva').

        Strict: surname ratio >= 0.90 AND first-name ratio >= 0.75 AND a
        single candidate. Sisters/homonyms safely fall back to default.
        """
        import difflib
        toks = name_tokens(name)
        if len(toks) < 2:
            return None
        q_first, q_surn = toks[0], "".join(toks[1:])
        if len(q_surn) < 4 or len(q_first) < 2:
            return None
        cands = []
        for surn, first, _key, player in self._fuzzy_index:
            if not surn or abs(len(surn) - len(q_surn)) > 2:
                continue
            if first[:1] != q_first[:1]:
                continue
            rs = difflib.SequenceMatcher(None, q_surn, surn).ratio()
            if rs < 0.90:
                continue
            rf = difflib.SequenceMatcher(None, q_first, first).ratio()
            if rf < 0.75:
                continue
            cands.append((rs, rf, player))
        if len(cands) == 1:
            _rs, _rf, player = cands[0]
            logger.info(f"Fuzzy match: '{name}' -> '{player.name}' "
                        f"(cognome {_rs:.2f}, nome {_rf:.2f})")
            return player
        return None

    def _resolve_player(self, player):
        """Return the historical profile for a live player, else the player itself.

        Match nuovo (nessuno storico) -> restituisce il Player originale con
        soli id/nome: il resto della pipeline prosegue con defaults
        invece di scartare il match.
        """
        if player is None:
            return None
        try:
            hit = self.player_db.get(player.id)
            if hit is not None:
                return hit
            key = normalize_name(player.name)
            hit = self._name_index.get(key)
            if hit is not None:
                return hit
            hit = self._fuzzy_player(player.name)
            if hit is not None:
                return hit
        except Exception:
            pass
        return player

    def _recent_store_path(self) -> Path:
        p = Path("data/recent/espn_results.json")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

    def _load_recent_store(self) -> List[Dict]:
        try:
            p = self._recent_store_path()
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
        except Exception as e:
            logger.warning(f"Recent store unreadable: {e}")
        return []

    def _save_recent_store(self, records: List[Dict]):
        try:
            self._recent_store_path().write_text(
                json.dumps(records, ensure_ascii=False, default=str),
                encoding="utf-8")
        except Exception as e:
            logger.warning(f"Recent store not saved: {e}")

    def _fetched_path(self) -> Path:
        p = Path("data/recent/espn_fetched.json")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

    def _load_fetched(self) -> set:
        try:
            p = self._fetched_path()
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return set(data)
        except Exception:
            pass
        return set()

    def _save_fetched(self, fetched: set) -> None:
        try:
            self._fetched_path().write_text(
                json.dumps(sorted(fetched), ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Fetched-dates not saved: {e}")

    def _backfill_history(self, days: int = 365) -> List[Dict]:
        """Scarica risultati ESPN giorno per giorno (ultimo anno).

        Date gia' presenti in espn_fetched.json saltate: primo avvio lungo,
        avvii successivi solo recupero giorni nuovi.
        """
        from datetime import date as _date, timedelta as _td
        import time as _time
        today = _date.today()
        wanted = [today - _td(days=b) for b in range(days - 1, -1, -1)]
        fetched = self._load_fetched()
        todo = [d for d in wanted if d.isoformat() not in fetched]
        logger.info(f"History backfill: {len(todo)} giorni da scaricare "
                    f"({len(fetched)} gia' in cache)")
        recent: List[Dict] = []
        if todo:
            progress.start(len(todo), "Storico 1 anno")
        for d in todo:
            try:
                progress.set_desc(f"Storico ESPN {d.isoformat()}")
                recent = self._update_recent_results(d)
                fetched.add(d.isoformat())
            except Exception as e:
                logger.warning(f"Recent-results {d} skipped: {e}")
            finally:
                progress.update()
            _time.sleep(0.3)  # gentile con le API ESPN
        if todo:
            progress.close()
            self._save_fetched(fetched)
        if not recent:
            recent = self._load_recent_store()
        # pota store oltre 400 giorni per non farlo crescere all'infinito
        try:
            if len(recent) > 20000:
                recent = sorted(recent, key=lambda r: str(r.get("date", "")))[-20000:]
                self._save_recent_store(recent)
        except Exception:
            pass
        return recent

    def _update_recent_results(self, target=None) -> List[Dict]:
        """Fetch ESPN completed matches and merge into the local store.

        Called at every startup (current week; cached 30min) and by the
        backfill script (past weeks). Returns the full store chronologically.
        Offline-safe: on failure returns what is stored.
        """
        from datetime import date as _date
        target = target or _date.today()
        store = self._load_recent_store()
        seen = {r.get("uid") for r in store if r.get("uid")}
        try:
            fresh = self.espn.get_results_for_date(target)
        except Exception as e:
            logger.warning(f"ESPN results fetch failed: {e}")
            fresh = []
        added = 0
        for r in fresh:
            if r.uid in seen:
                continue
            seen.add(r.uid)
            store.append({
                "uid": r.uid, "tournament": r.tournament, "league": r.league,
                "round": r.round, "best_of": r.best_of, "surface": r.surface,
                "level": r.level, "winner": r.winner, "loser": r.loser,
                "date": r.date.isoformat() if hasattr(r.date, "isoformat") else str(r.date),
            })
            added += 1
        if added:
            self._save_recent_store(store)
            logger.info(f"Recent results: +{added} nuovi (store: {len(store)})")
        store.sort(key=lambda r: str(r.get("date", "")))
        return store

    def _apply_recent_results(self, records: List[Dict]):
        """Replay recent results onto profiles, H2H index and history.

        Same incremental math as build_player_profiles (win%, surface Elo
        k=32, last-10 form), so current form (e.g. summer titles) is priced.
        Unknown names become espn_-prefixed bare profiles (no crash, the
        per-side fallback still applies downstream).
        """
        from src.utils.models import Player as _Player
        k_factor = 32.0
        n = 0
        # Dedup vs snapshot: RG/Wimbledon overlap (stesso match, nomi diversi
        # ESPN/Sackmann) -> skip se la coppia ha un match entro ±10 giorni.
        pair_dates: Dict[frozenset, List[int]] = {}
        for m in self.historical_matches:
            try:
                a, b = str(m.get("winner_id", "")), str(m.get("loser_id", ""))
                if not a or not b:
                    continue
                d = int(str(m.get("tourney_date", "0"))[:8])
                pair_dates.setdefault(frozenset((a, b)), []).append(d)
            except (ValueError, TypeError):
                continue
        n_dup = 0
        for r in records:
            try:
                wname, lname = r.get("winner", ""), r.get("loser", "")
                if not wname or not lname:
                    continue
                wid = self._player_key(wname)
                lid = self._player_key(lname)
                try:
                    ed = int(str(r.get("date", ""))[:10].replace("-", ""))
                except (ValueError, TypeError):
                    ed = 0
                if any(abs(d - ed) <= 10 for d in pair_dates.get(frozenset((wid, lid)), [])):
                    n_dup += 1
                    continue  # già nello snapshot
                pair_dates.setdefault(frozenset((wid, lid)), []).append(ed)
                surface = self.sackmann.parse_surface(r.get("surface", "Hard"))
                for pid, pname in ((wid, wname), (lid, lname)):
                    if pid not in self.player_db:
                        self.player_db[pid] = _Player(id=pid, name=str(pname))
                wpl, lpl = self.player_db[wid], self.player_db[lid]
                for pl, won in ((wpl, True), (lpl, False)):
                    if surface not in pl.surface_matches:
                        pl.surface_matches[surface] = 0
                        pl.surface_win_pct[surface] = 0.0
                        pl.surface_elo[surface] = 1500.0
                    pl.surface_matches[surface] += 1
                    cw = pl.surface_win_pct[surface] * (pl.surface_matches[surface] - 1)
                    if won:
                        cw += 1
                    pl.surface_win_pct[surface] = cw / pl.surface_matches[surface]
                    buf = getattr(pl, "_recent_results", None)
                    if buf is None:
                        buf = [1] * pl.wins_last_10 + [0] * pl.losses_last_10
                    buf.append(1 if won else 0)
                    if len(buf) > 10:
                        buf = buf[-10:]
                    pl._recent_results = buf  # type: ignore[attr-defined]
                    pl.wins_last_10 = sum(buf)
                    pl.losses_last_10 = len(buf) - pl.wins_last_10
                we = wpl.surface_elo.get(surface, 1500.0)
                le = lpl.surface_elo.get(surface, 1500.0)
                ew = 1.0 / (1.0 + 10 ** ((le - we) / 400.0))
                wpl.surface_elo[surface] = we + k_factor * (1 - ew)
                lpl.surface_elo[surface] = le + k_factor * (0 - (1 - ew))
                # H2H index (ESPN full names, as used by live matching)
                key = tuple(sorted([str(wname), str(lname)]))
                self._h2h_index.setdefault(key, []).append({
                    "winner": wname, "loser": lname,
                    "surface": surface.value if hasattr(surface, "value") else str(surface),
                    "date": str(r.get("date", "")).replace("-", "")[:8],
                    "score": "",
                })
                self.h2h_cache.pop(key, None)
                # history dict for the ML training set (Sackmann-shaped)
                raw_date = str(r.get("date", ""))[:10].replace("-", "")
                self.historical_matches.append({
                    "winner_id": wid, "loser_id": lid,
                    "winner_name": wname, "loser_name": lname,
                    "surface": surface.value if hasattr(surface, "value") else str(surface),
                    "tourney_level": self._espn_level_code(r.get("level", "")),
                    "tour": str(r.get("league", "atp")),
                    "tourney_name": r.get("tournament", "Recent"),
                    "tourney_date": raw_date if len(raw_date) == 8 else "20260101",
                    "round": r.get("round", "R64"),
                })
                n += 1
            except Exception:
                continue
        # finalize temp form buffers
        for pl in self.player_db.values():
            if hasattr(pl, "_recent_results"):
                try:
                    delattr(pl, "_recent_results")
                except AttributeError:
                    pass
        logger.info(f"Replayed {n} recent results onto profiles/H2H/history "
                    f"({n_dup} duplicati snapshot saltati)")

    def _player_key(self, name: str) -> str:
        """Stable id for a live name: Sackmann id when known, else espn_ key."""
        try:
            hit = self.player_db.get(name)
            if hit is not None:
                return hit.id
            hit = self._name_index.get(normalize_name(name))
            if hit is not None:
                return hit.id
            hit = self._fuzzy_player(name)
            if hit is not None:
                return hit.id
        except Exception:
            pass
        return f"espn_{normalize_name(name)}"

    @staticmethod
    def _espn_level_code(level_hint: str) -> str:
        return {"GS": "G", "MS1000": "M", "WTA1000": "M",
                "ATP500": "A", "WTA500": "A", "ATP250": "B", "WTA250": "B",
                "WTA125": "C", "CH100": "C"}.get(str(level_hint or ""), "")

    def _build_h2h_index(self):
        """Pre-index historical matches by player pair for fast H2H lookup."""
        self.h2h_cache = {}
        self._h2h_index: Dict[tuple, List[Dict]] = {}
        for match in self.historical_matches:
            try:
                winner = match.get('winner_name', '')
                loser = match.get('loser_name', '')
                if not winner or not loser:
                    continue
                key = tuple(sorted([str(winner), str(loser)]))
                self._h2h_index.setdefault(key, []).append({
                    'winner': winner,
                    'loser': loser,
                    'surface': match.get('surface', 'Hard'),
                    'date': match.get('tourney_date', ''),
                    'score': match.get('score', '')
                })
            except Exception:
                continue
        logger.info(f"H2H index built: {len(self._h2h_index)} pairs")

    def _get_poly_sentiment(self, matches: List[TennisMatch],
                              demo: bool = False) -> Dict[str, tuple]:
        """Polymarket crowd probability per match, oriented (p1, p2).

        In demo sandbox returns {} (no invented sentiment). Misses simply
        yield no entry -> the ensemble weight stays unused for that match.
        """
        if demo or not matches:
            return {}
        try:
            from src.data.ingest.polymarket import PolymarketClient
            if self.polymarket is None:
                self.polymarket = PolymarketClient()
            out: Dict[str, tuple] = {}
            for m in matches:
                if not m.player1 or not m.player2:
                    continue
                try:
                    found = self.polymarket._find_h2h(m.player1.name, m.player2.name)
                except Exception:
                    found = None
                if found:
                    out[m.id] = (float(found[0]), float(found[1]))
            logger.info(f"Polymarket sentiment: {len(out)}/{len(matches)} match coperti")
            return out
        except Exception as e:
            logger.warning(f"Polymarket sentiment failed: {e}")
            return {}

    def _build_features(self, matches: List[TennisMatch],
                        odds_dict: Dict[str, MergedOdds],
                        news_impact: Dict,
                        poly_dict: Dict[str, tuple] = None) -> Dict[str, TennisFeatures]:
        """Build features for all matches"""
        features = {}

        from dataclasses import replace as _replace

        progress.start(len(matches), "Building features")
        for match in matches:
            merged_odds = odds_dict.get(match.id)
            poly = (poly_dict or {}).get(match.id)
            # Resolve live names -> historical profiles when available;
            # new players keep their bare Player and take the default path.
            try:
                r1 = self._resolve_player(match.player1)
                r2 = self._resolve_player(match.player2)
                rmatch = _replace(match, player1=r1, player2=r2)
            except Exception:
                rmatch = match
            h2h = None
            if rmatch.player1 and rmatch.player2:
                h2h = self._get_h2h(rmatch.player1.name, rmatch.player2.name)

            feat = self.feature_builder.build_features(
                match=rmatch,
                player_db=self.player_db,
                merged_odds=merged_odds,
                news_impact=news_impact,
                h2h_data=h2h,
                poly=poly
            )

            features[match.id] = feat
            progress.update()

        progress.close()
        return features

    def _get_h2h(self, player1: str, player2: str) -> List[Dict]:
        """Get head-to-head history (indexed, with legacy fallback)."""
        key = tuple(sorted([str(player1), str(player2)]))
        if key in self.h2h_cache:
            return self.h2h_cache[key]

        # Fast path: pre-built index
        index = getattr(self, '_h2h_index', None)
        if index is not None:
            h2h = index.get(key, [])
            self.h2h_cache[key] = h2h
            return h2h

        h2h = []
        for match in self.historical_matches:
            winner = match.get('winner_name', '')
            loser = match.get('loser_name', '')
            if (winner == player1 and loser == player2) or (winner == player2 and loser == player1):
                h2h.append({
                    'winner': winner,
                    'loser': loser,
                    'surface': match.get('surface', 'Hard'),
                    'date': match.get('tourney_date', ''),
                    'score': match.get('score', '')
                })

        self.h2h_cache[key] = h2h
        return h2h
    
    def _fit_models(self, max_baseline_matches: int = 60000):
        """Fit ensemble models on historical data (capped for speed)."""
        logger.info("Fitting models...")

        hist = self.historical_matches
        if len(hist) > max_baseline_matches:
            # Most recent first for demo/training relevance, then back to
            # chronological for Elo-style fitting.
            hist_sorted = sorted(hist, key=lambda m: str(m.get('tourney_date', '19000101')))
            hist = hist_sorted[-max_baseline_matches:]
            logger.info(f"Capped baseline fitting to most recent {len(hist)} matches")

        self.ensemble.fit(
            historical_matches=hist,
            players=self.player_db,
            features_list=self.training_features,
            labels=self.training_labels
        )

        self._models_fitted = True
        logger.info("Models fitted")
    
    def _generate_predictions(self, features_dict: Dict[str, TennisFeatures]) -> Dict[str, TennisPrediction]:
        """Generate predictions for all matches"""
        predictions = {}

        progress.start(len(features_dict), "Generating predictions")
        for match_id, features in features_dict.items():
            pred = self.ensemble.predict(features)
            predictions[match_id] = pred
            progress.update()

        progress.close()
        return predictions
    
    def _find_value_bets(self, predictions: Dict[str, TennisPrediction],
                         odds_dict: Dict[str, MergedOdds],
                         matches: List[TennisMatch]) -> List[ValueBet]:
        """Find value bets"""
        match_info = {}
        for m in matches:
            match_info[m.id] = {
                'match_key': m.match_key,
                'tournament': m.tournament,
                'player1': m.player1.name if m.player1 else '',
                'player2': m.player2.name if m.player2 else '',
            }
        
        pred_list = list(predictions.values())
        value_bets = self.value_detector.find_all_value(pred_list, odds_dict, match_info)
        
        return value_bets
    
    def _select_decisions(self, value_bets: List[ValueBet],
                            features_dict: Dict[str, TennisFeatures],
                            odds_dict: Dict[str, MergedOdds]):
        """Final decisions: dedup filters only, no money involved.

        Returns (decisions, excluded) with reason + Sisal coverage each.
        """
        bet_cfg = self.config.get('betting', {}) if isinstance(self.config, dict) else {}
        day_cap = config.get('risk', 'max_bets_per_day', default=10)
        tourney_cap = bet_cfg.get('max_bets_per_tournament_day', 3)
        player_cap = bet_cfg.get('max_bets_per_player_week', 2)

        decisions: List[BetDecision] = []
        excluded: List[BetDecision] = []
        tourney_counts: Dict[str, int] = {}
        player_counts: Dict[str, int] = {}

        def _decorate(bet: ValueBet) -> BetDecision:
            feat = features_dict.get(bet.match_id)
            reason = explain_value_bet(bet, feat)
            odds = odds_dict.get(bet.match_id)
            flag, note = self._sisal_presence_bet(bet, odds)
            if note:
                reason = f"{reason}; Sisal: {note}"
            book = ""
            if odds:
                book = (odds.best_book_home if bet.side == "player1"
                        else odds.best_book_away)
            return BetDecision(value_bet=bet, reason=reason,
                               bookmaker=book or "?", sisal=flag)

        for bet in sorted(value_bets, key=lambda b: b.confidence * b.edge,
                          reverse=True):
            dec = _decorate(bet)
            tq = bet.tournament or "?"
            if len(decisions) >= day_cap:
                dec.excluded_reason = f"limite giornaliero ({day_cap} decisioni)"
                excluded.append(dec)
                continue
            if tourney_counts.get(tq, 0) >= tourney_cap:
                dec.excluded_reason = (f"torneo '{tq}' già con "
                                       f"{tourney_cap} decisioni")
                excluded.append(dec)
                continue
            if player_counts.get(bet.player_name, 0) >= player_cap:
                dec.excluded_reason = (f"{bet.player_name} già con "
                                       f"{player_cap} decisioni")
                excluded.append(dec)
                continue
            dec.rank = len(decisions) + 1
            decisions.append(dec)
            tourney_counts[tq] = tourney_counts.get(tq, 0) + 1
            player_counts[bet.player_name] = player_counts.get(bet.player_name, 0) + 1

        return decisions, excluded

    def _generate_report(self, target_date: date, result: PipelineResult):
        """Generate daily report (decisions only, no money)."""
        report = {
            'date': target_date.isoformat(),
            'mode': 'DEMO' if result.demo_mode else 'LIVE',
            'analyzed_events': [],
        }
        for m in result.matches:
            pred = result.predictions.get(m.id)
            odds = result.odds.get(m.id)
            if not pred:
                continue
            report['analyzed_events'].append({
                'match': m.match_key,
                'tournament': m.tournament,
                'surface': m.surface.value if hasattr(m.surface, 'value') else str(m.surface),
                'model_prob_p1': round(pred.p1_win_prob, 4),
                'model_prob_p2': round(pred.p2_win_prob, 4),
                'market_prob_p1': round(odds.implied_prob_home, 4) if odds else None,
                'best_odds_home': odds.best_odds_home if odds else None,
                'best_odds_away': odds.best_odds_away if odds else None,
                'tier': pred.tier.value if hasattr(pred.tier, 'value') else pred.tier,
                'confidence': round(pred.confidence, 3),
                'data_quality': pred.data_quality_flag,
                'sisal': (result.sisal_flag.get(m.id, ("?", ""))[0]
                          if hasattr(result, "sisal_flag") else "?"),
            })
        
        # Today's decisions (with reasons, no stakes)
        def _dec_json(d: BetDecision):
            vb = d.value_bet
            return {
                'match': vb.match_key,
                'player': vb.player_name,
                'side': vb.side,
                'odds': vb.odds,
                'bookmaker': d.bookmaker,
                'edge': round(vb.edge, 4),
                'model_prob': round(vb.model_prob, 4),
                'tier': vb.tier.value if hasattr(vb.tier, 'value') else vb.tier,
                'confidence': round(vb.confidence, 3),
                'sisal': d.sisal,
                'reason': d.reason,
                'rank': d.rank,
                'excluded_reason': d.excluded_reason,
            }

        report['decisions'] = [_dec_json(d) for d in result.decisions]
        report['excluded'] = [_dec_json(d) for d in result.excluded]
        report['summary'] = result.decision_summary
        
        # Save detailed report
        output_dir = Path(self.app_config.get('output_dir', 'output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / f"pipeline_report_{target_date.isoformat()}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved: {report_file}")


async def run_daily_pipeline(target_date: date = None,
                             demo_mode: bool = False) -> PipelineResult:
    """Convenience function to run daily pipeline"""
    pipeline = TennisDailyPipeline(demo_mode)
    return await pipeline.run(target_date)