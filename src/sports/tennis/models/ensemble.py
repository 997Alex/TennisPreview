"""Ensemble model combining baseline, ML, and market predictions"""
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from src.utils.config import config
from src.utils.logging import get_logger
from src.utils.models import (
    TennisFeatures, TennisPrediction, DataTier, Player, TennisMatch
)
from src.sports.tennis.models.baseline import (
    BradleyTerryModel, Glicko2Model, EloFormModel
)
from src.sports.tennis.models.ml_models import (
    TennisXGBoostModel, TennisLightGBMModel, SurfaceLogisticModel
)

logger = get_logger("tennis_ensemble")


class TennisEnsemble:
    """Ensemble predictor with tier-aware model weighting"""
    
    def __init__(self):
        self.config = config.get('tennis', 'models')
        
        # Model weights per tier
        self.weights_tier1 = self.config.get('ensemble_weights_tier1', {})
        self.weights_tier2 = self.config.get('ensemble_weights_tier2', {})
        self.weights_tier3 = self.config.get('ensemble_weights_tier3', {})
        
        # Models
        self.baseline_models = {}
        self.ml_models = {}
        self.calibrators = {}
        
        self.is_fitted = False
        self.training_history = []
    
    def initialize_models(self):
        """Initialize all sub-models"""
        self.baseline_models = {
            'bradley_terry': BradleyTerryModel(),
            'glicko2': Glicko2Model(),
            'elo_form': EloFormModel(),
        }
        
        ml_config = {
            'xgb_params': {},
            'n_estimators': 500,
            'early_stopping_rounds': 50,
        }
        
        self.ml_models = {
            'xgboost': TennisXGBoostModel(ml_config),
            'surface_logistic': SurfaceLogisticModel(),
        }
    
    def fit(self, historical_matches: List[Dict], 
            players: Dict[str, Player],
            features_list: List[TennisFeatures],
            labels: List[int]):
        """Fit all models on historical data"""
        logger.info(f"Fitting ensemble on {len(historical_matches)} matches")
        
        self.initialize_models()
        
        # Fit baseline models
        logger.info("Fitting baseline models...")
        for name, model in self.baseline_models.items():
            try:
                model.fit(historical_matches, players)
                logger.info(f"  {name} fitted")
            except Exception as e:
                logger.error(f"  {name} failed: {e}")
        
        # Fit ML models (only if training features provided)
        if features_list and labels:
            logger.info("Fitting ML models...")
            for name, model in self.ml_models.items():
                try:
                    model.fit(features_list, labels)
                    logger.info(f"  {name} fitted")
                except Exception as e:
                    logger.error(f"  {name} failed: {e}")
            
            # Fit calibrators per tier
            self._fit_calibrators(features_list, labels)
        else:
            logger.warning("No training features provided - skipping ML fitting")
        
        self.is_fitted = True
        logger.info("Ensemble fitting complete")
    
    @staticmethod
    def _market_present(features: TennisFeatures) -> bool:
        """True only with real merged odds (best_odds set ⇔ merged_odds passed).

        Training features and odds-less matches carry best_odds = 0.0, so the
        market component is structurally absent there.
        """
        try:
            return float(features.best_odds_home) > 1.0
        except (TypeError, ValueError):
            return False

    def _tier_weights(self, tier: DataTier):
        if tier == DataTier.TIER_1:
            return self.weights_tier1
        elif tier == DataTier.TIER_2:
            return self.weights_tier2
        return self.weights_tier3

    def _fit_calibrators(self, features_list: List[TennisFeatures], labels: List[int]):
        """Fit calibration models per tier on NO-MARKET blends.

        Training features never carry market odds, so calibrating the
        market-inclusive blend would learn on a different distribution than
        inference (extrapolation at the extremes). The market is fused
        AFTER calibration (see predict).
        """
        from sklearn.isotonic import IsotonicRegression

        for tier in [DataTier.TIER_1, DataTier.TIER_2, DataTier.TIER_3]:
            tier_features = [f for f in features_list if f.data_tier == tier]
            tier_labels = [labels[i] for i, f in enumerate(features_list) if f.data_tier == tier]

            if len(tier_features) > 100:
                # Get ensemble predictions for this tier (no market/poly:
                # absent at fit time, fused post-calibration at inference)
                preds = []
                for f in tier_features:
                    pred = self._predict_uncalibrated(f, include_market=False,
                                                      include_poly=False)
                    preds.append(pred.p1_win_prob)

                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(preds, tier_labels)
                self.calibrators[tier] = calibrator
                logger.info(f"Calibrator fitted for {tier.name}: {len(tier_features)} samples")
    
    @staticmethod
    def _poly_present(features: TennisFeatures) -> bool:
        """True only with a real Polymarket price (0.0 = assente)."""
        try:
            return 0.0 < float(features.poly_implied_p1) < 1.0
        except (TypeError, ValueError):
            return False

    def _predict_uncalibrated(self, features: TennisFeatures,
                                include_market: bool = True,
                                include_poly: bool = True) -> TennisPrediction:
        """Get raw ensemble prediction without calibration"""
        tier = features.data_tier

        # Build weight dict keyed by model output name
        active_weights = self._active_weights(features)

        model_probs = {}
        
        # Baseline models (need KNOWN player IDs: unknown ids would vote
        # a meaningless 0.5 and dilute the informed side)
        if 'bradley_terry' in active_weights and 'bradley_terry' in self.baseline_models:
            bt = self.baseline_models['bradley_terry']
            if (features.player1_id in bt.ratings
                    and features.player2_id in bt.ratings):
                model_probs['bradley_terry'] = bt.predict(
                    features.player1_id, features.player2_id, features.surface
                )

        if 'glicko2' in active_weights and 'glicko2' in self.baseline_models:
            glicko = self.baseline_models['glicko2']
            if (features.player1_id in glicko.ratings
                    and features.player2_id in glicko.ratings):
                model_probs['glicko2'] = glicko.predict(
                    features.player1_id, features.player2_id, features.surface
                )
        
        if 'elo_form' in active_weights and 'elo_form' in self.baseline_models:
            model_probs['elo_form'] = self.baseline_models['elo_form'].predict(features)
        
        # ML models
        if 'xgboost' in active_weights and 'xgboost' in self.ml_models:
            xgb = self.ml_models['xgboost']
            if getattr(xgb, 'is_fitted', False):
                model_probs['xgboost'] = xgb.predict_proba(features)
        
        if 'surface_logistic' in active_weights and 'surface_logistic' in self.ml_models:
            model_probs['surface_logistic'] = self.ml_models['surface_logistic'].predict_proba(features)
        
        # Market implied: only when requested AND real odds present.
        # (Training/odds-less features carry market 0.5 by default, which
        # must not anchor the blend.)
        market_here = (include_market and self._market_present(features)
                       and features.market_implied_p1 > 0)
        if 'market' in active_weights and market_here:
            model_probs['market'] = features.market_implied_p1

        # Polymarket sentiment: same present/absent discipline as market.
        poly_here = (include_poly and self._poly_present(features))
        if 'polymarket' in active_weights and poly_here:
            model_probs['polymarket'] = features.poly_implied_p1
        
        # H2H
        if 'h2h' in active_weights and features.h2h_total > 0:
            model_probs['h2h'] = features.h2h_p1_wins / features.h2h_total
        
        # Weighted average
        if not model_probs:
            p1_prob = features.market_implied_p1 if features.market_implied_p1 > 0 else 0.5
        else:
            total_weight = sum(active_weights.get(k, 0.0) for k in model_probs)
            if total_weight > 0:
                p1_prob = sum(model_probs[k] * active_weights.get(k, 0.0) for k in model_probs) / total_weight
            else:
                p1_prob = float(np.mean(list(model_probs.values())))
        
        p1_prob = float(np.clip(p1_prob, 0.01, 0.99))
        
        pred = TennisPrediction(
            match_id=features.match_id,
            p1_win_prob=p1_prob,
            p2_win_prob=1 - p1_prob,
            model_probs=model_probs,
            tier=tier,
            confidence=0.5,
            data_quality_flag=self._quality_flag(features),
            bradley_terry_prob=model_probs.get('bradley_terry', 0),
            glicko_prob=model_probs.get('glicko2', 0),
            xgb_prob=model_probs.get('xgboost', 0),
            elo_form_prob=model_probs.get('elo_form', 0),
            market_prob=model_probs.get('market', 0)
        )
        
        pred.confidence = self._calculate_confidence(tier, features, pred)
        return pred
    
    def predict(self, features: TennisFeatures) -> TennisPrediction:
        """Make calibrated prediction"""
        if not self.is_fitted:
            # Return market probability if not fitted
            return TennisPrediction(
                match_id=features.match_id,
                p1_win_prob=features.market_implied_p1,
                p2_win_prob=features.market_implied_p2,
                model_probs={'market': features.market_implied_p1},
                tier=features.data_tier,
                confidence=0.3,
                data_quality_flag="model_not_fitted"
            )
        
        # 1) No-market blend. NOTE: isotonic calibrators are DISABLED in
        #    live predictions. Backtests on 684 Slam matches with real odds
        #    (USO25/AO26/RG26 x ATP/WTA) showed them net-negative pooled:
        #    Brier .2045 vs .2025 raw, logloss .608 vs .593, ECE .104 vs .090
        #    (worse in 5/6 events: time-regime shift train->test overfits the
        #    non-parametric map). _fit_calibrators stays for research/sweep.
        raw = self._predict_uncalibrated(features, include_market=False)
        p_nomkt = raw.p1_win_prob

        # 2) Late fusion of crowd extras (market odds, Polymarket sentiment)
        #    with config weights, only when really present. Without extras,
        #    or without any model, the blend degrades honestly.
        active = self._active_weights(features)
        extras = []
        if self._market_present(features) and features.market_implied_p1 > 0:
            extras.append((active.get("market", 0.0), features.market_implied_p1))
        if self._poly_present(features):
            extras.append((active.get("polymarket", 0.0), features.poly_implied_p1))
        rest_w = sum(w for k, w in active.items()
                     if k not in ("market", "polymarket") and k in raw.model_probs)

        if rest_w > 0:
            num = rest_w * p_nomkt + sum(w * p for w, p in extras if w > 0)
            den = rest_w + sum(w for w, p in extras if w > 0)
            p_final = num / den if den > 0 else p_nomkt
        elif extras:
            tot = sum(w for w, p in extras if w > 0)
            p_final = (sum(w * p for w, p in extras if w > 0) / tot) if tot > 0 else 0.5
        else:
            p_final = 0.5
        p_final = float(np.clip(p_final, 0.01, 0.99))

        # Full breakdown for display (market included when present).
        pred = self._predict_uncalibrated(features, include_market=True)
        pred.p1_win_prob = p_final
        pred.p2_win_prob = 1 - p_final
        pred.confidence = self._calculate_confidence(features.data_tier, features, pred)
        return pred

    def _active_weights(self, features: TennisFeatures) -> Dict[str, float]:
        """Config weights keyed by model output name for the feature tier."""
        key_mapping = {
            'bradley_terry_surface': 'bradley_terry',
            'glicko2_surface': 'glicko2',
            'serve_return_xgb': 'xgboost',
            'basic_xgb': 'xgboost',
            'elo_form': 'elo_form',
            'market_implied': 'market',
            'polymarket': 'polymarket',
            'surface_win_pct_logistic': 'surface_logistic',
            'h2h_logistic': 'h2h',
        }
        weights = self._tier_weights(features.data_tier)
        active = {}
        for config_key, weight in weights.items():
            model_key = key_mapping.get(config_key)
            if model_key and weight > 0:
                active[model_key] = active.get(model_key, 0.0) + weight
        return active
    
    def predict_batch(self, features_list: List[TennisFeatures]) -> List[TennisPrediction]:
        """Predict for multiple matches"""
        return [self.predict(f) for f in features_list]
    
    def _calculate_confidence(self, tier: DataTier, features: TennisFeatures,
                              pred: TennisPrediction) -> float:
        """Calculate prediction confidence"""
        base_confidence = {
            DataTier.TIER_1: 0.85,
            DataTier.TIER_2: 0.70,
            DataTier.TIER_3: 0.50,
            DataTier.TIER_4: 0.20,
        }[tier]
        
        # Adjust for feature completeness
        confidence = base_confidence * (0.5 + 0.5 * features.feature_completeness)

        # Debutant on either side: less trust in the blend
        if getattr(features, "partial_profile", False):
            confidence *= 0.8
        
        # Adjust for market agreement (only with real odds; the 0.5
        # default of odds-less features must not move confidence)
        if self._market_present(features) and features.market_implied_p1 > 0:
            market_edge = abs(pred.p1_win_prob - features.market_implied_p1)
            if market_edge < 0.05:
                confidence *= 1.1
            elif market_edge > 0.15:
                confidence *= 0.8
        
        return float(np.clip(confidence, 0.1, 0.95))
    
    def _quality_flag(self, features: TennisFeatures) -> str:
        """Generate data quality flag"""
        flags = []
        
        if features.data_tier == DataTier.TIER_4:
            flags.append("MARKET_ONLY")
        elif features.data_tier == DataTier.TIER_3:
            flags.append("LIMITED_DATA")
        
        if features.feature_completeness < 0.5:
            flags.append("LOW_COMPLETENESS")
        
        if features.h2h_total == 0:
            flags.append("NO_H2H")
        
        if getattr(features, "partial_profile", False):
            flags.append("PARTIAL_PROFILE")

        if features.p1_surface_win_pct == 0.5 and features.p2_surface_win_pct == 0.5:
            flags.append("NO_SURFACE_HISTORY")
        
        return "|".join(flags) if flags else "OK"


def get_tennis_ensemble() -> TennisEnsemble:
    return TennisEnsemble()