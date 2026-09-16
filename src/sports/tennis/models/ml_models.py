"""ML models for tennis prediction (XGBoost, LightGBM)"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pickle
import json

from src.utils.config import config
from src.utils.logging import get_logger
from src.utils.models import TennisFeatures, TennisPrediction, DataTier

logger = get_logger("tennis_ml_models")

XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError) as e:
    logger.warning(f"LightGBM not available: {e}")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class TennisXGBoostModel:
    """XGBoost model for tennis match prediction"""
    
    def __init__(self, config_dict: Dict = None):
        self.config = config_dict or {}
        self.model = None
        self.calibrator = None
        self.feature_names = None
        self.is_fitted = False
        
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': 0.03,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'tree_method': 'hist',
            'random_state': 42,
            'nthread': -1,
        }
        self.params.update(self.config.get('xgb_params', {}))
        
        self.n_estimators = self.config.get('n_estimators', 500)
        self.early_stopping = self.config.get('early_stopping_rounds', 50)
    
    def prepare_data(self, features_list: List[TennisFeatures], 
                     labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert features to arrays"""
        X = np.array([f.to_array() for f in features_list])
        y = np.array(labels)
        return X, y
    
    def fit(self, features_list: List[TennisFeatures], 
            labels: List[int],
            val_features: List[TennisFeatures] = None,
            val_labels: List[int] = None):
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost not installed")
        
        X, y = self.prepare_data(features_list, labels)
        self.feature_names = features_list[0].feature_names if features_list else []
        
        # Validation split
        if val_features is None or val_labels is None:
            # Time series split - use last 20% as validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y
            X_val, y_val = self.prepare_data(val_features, val_labels)
        
        logger.info(f"Training XGBoost on {len(X_train)} samples, validating on {len(X_val)}")
        
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping,
            verbose_eval=100
        )
        
        # Calibrate
        val_preds = self.model.predict(dval)
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(val_preds, y_val)
        
        self.is_fitted = True
        logger.info("XGBoost training complete")
        
        # Feature importance
        importance = self.model.get_score(importance_type='gain')
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Top 10 features: {sorted_imp[:10]}")
    
    def predict_proba(self, features: TennisFeatures) -> float:
        """Predict win probability for player 1"""
        if not self.is_fitted:
            return 0.5
        
        X = np.array([features.to_array()])
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        raw_pred = self.model.predict(dtest)[0]
        
        # Calibrate
        calibrated = self.calibrator.predict([raw_pred])[0]
        return np.clip(calibrated, 0.01, 0.99)
    
    def predict_batch(self, features_list: List[TennisFeatures]) -> np.ndarray:
        """Predict for multiple matches"""
        if not self.is_fitted or not features_list:
            return np.full(len(features_list), 0.5)
        
        X = np.array([f.to_array() for f in features_list])
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        raw_preds = self.model.predict(dtest)
        calibrated = self.calibrator.predict(raw_preds)
        return np.clip(calibrated, 0.01, 0.99)
    
    def save(self, path: str):
        """Save model"""
        if self.model:
            self.model.save_model(f"{path}.xgb")
            with open(f"{path}_calibrator.pkl", 'wb') as f:
                pickle.dump(self.calibrator, f)
            with open(f"{path}_meta.json", 'w') as f:
                json.dump({
                    'feature_names': self.feature_names,
                    'params': self.params
                }, f)
    
    def load(self, path: str):
        """Load model"""
        if XGBOOST_AVAILABLE:
            self.model = xgb.Booster()
            self.model.load_model(f"{path}.xgb")
            with open(f"{path}_calibrator.pkl", 'rb') as f:
                self.calibrator = pickle.load(f)
            with open(f"{path}_meta.json", 'r') as f:
                meta = json.load(f)
                self.feature_names = meta['feature_names']
                self.params = meta['params']
            self.is_fitted = True


class TennisLightGBMModel:
    """LightGBM model for tennis prediction"""
    
    def __init__(self, config_dict: Dict = None):
        self.config = config_dict or {}
        self.model = None
        self.calibrator = None
        self.feature_names = None
        self.is_fitted = False
        
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 63,
            'max_depth': 6,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }
        self.params.update(self.config.get('lgb_params', {}))
        
        self.n_estimators = self.config.get('n_estimators', 500)
        self.early_stopping = self.config.get('early_stopping_rounds', 50)
    
    def fit(self, features_list: List[TennisFeatures], 
            labels: List[int],
            val_features: List[TennisFeatures] = None,
            val_labels: List[int] = None):
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("LightGBM not installed")
        
        X, y = np.array([f.to_array() for f in features_list]), np.array(labels)
        self.feature_names = features_list[0].feature_names if features_list else []
        
        if val_features is None:
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y
            X_val, y_val = np.array([f.to_array() for f in val_features]), np.array(val_labels)
        
        logger.info(f"Training LightGBM on {len(X_train)} samples")
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names)
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(self.early_stopping),
                lgb.log_evaluation(100)
            ]
        )
        
        # Calibrate
        val_preds = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(val_preds, y_val)
        
        self.is_fitted = True
        logger.info("LightGBM training complete")
    
    def predict_proba(self, features: TennisFeatures) -> float:
        if not self.is_fitted:
            return 0.5
        
        X = np.array([features.to_array()])
        raw = self.model.predict(X, num_iteration=self.model.best_iteration)[0]
        calibrated = self.calibrator.predict([raw])[0]
        return np.clip(calibrated, 0.01, 0.99)
    
    def predict_batch(self, features_list: List[TennisFeatures]) -> np.ndarray:
        if not self.is_fitted or not features_list:
            return np.full(len(features_list), 0.5)
        
        X = np.array([f.to_array() for f in features_list])
        raw = self.model.predict(X, num_iteration=self.model.best_iteration)
        calibrated = self.calibrator.predict(raw)
        return np.clip(calibrated, 0.01, 0.99)


class SurfaceLogisticModel:
    """Simple logistic regression per surface (for Tier 3)"""
    
    def __init__(self):
        self.models = {surface: LogisticRegression(max_iter=1000) for surface in ['Hard', 'Clay', 'Grass']}
        self.is_fitted = False
    
    def fit(self, features_list: List[TennisFeatures], labels: List[int]):
        """Fit separate model per surface"""
        surface_data = {'Hard': ([], []), 'Clay': ([], []), 'Grass': ([], [])}
        
        for f, label in zip(features_list, labels):
            surf = f.surface.value if hasattr(f.surface, 'value') else str(f.surface)
            if surf in surface_data:
                surface_data[surf][0].append(f.to_array())
                surface_data[surf][1].append(label)
        
        for surface, (X_list, y_list) in surface_data.items():
            if len(X_list) > 50:
                X = np.array(X_list)
                y = np.array(y_list)
                self.models[surface].fit(X, y)
        
        self.is_fitted = True
        logger.info("Surface logistic models fitted")
    
    def predict_proba(self, features: TennisFeatures) -> float:
        if not self.is_fitted:
            return 0.5
        
        surf = features.surface.value if hasattr(features.surface, 'value') else str(features.surface)
        if surf not in self.models:
            surf = 'Hard'
        
        X = np.array([features.to_array()])
        try:
            return self.models[surf].predict_proba(X)[0, 1]
        except:
            return 0.5


def get_ml_models(config_dict: Dict = None) -> Dict:
    """Get all ML models"""
    models = {}
    
    if XGBOOST_AVAILABLE:
        models['xgboost'] = TennisXGBoostModel(config_dict)
    
    if LIGHTGBM_AVAILABLE:
        models['lightgbm'] = TennisLightGBMModel(config_dict)
    
    models['surface_logistic'] = SurfaceLogisticModel()
    
    return models