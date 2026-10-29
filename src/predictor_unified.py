"""
predictor_unified.py - LOCAL VERSION CHO VSCODE
✅ Load models từ thư mục models/ (local)
✅ Tự động tìm sub-bundles trong cùng folder
✅ Tương thích với NSGA-II optimization
"""
import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Optional

# =========================================================================
# BƯỚC 1: DEFINE OptimalSlumpFeatureBuilder (cho pickle namespace)
# =========================================================================
class OptimalSlumpFeatureBuilder:
    """Feature builder cho slump model"""
    def __init__(self):
        self.feature_names = [
            'cement', 'water', 'fine_agg', 'coarse_agg', 'sp',
            'fly_ash', 'slag', 'silica_fume',
            'binder', 'w_b', 'scm_frac', 'sand_ratio', 'sp_per_b', 'sp_per_w',
            'paste_volume', 'agg_total', 'paste_to_agg', 'effective_w_c', 'pozzolanic_idx',
            'w_b_x_scm', 'w_b_x_sp', 'sp_x_scm', 'w_b_sq', 'sp_per_b_sq',
            'log_sp', 'log_silica_fume',
            'sp_saturation', 'excess_water_idx', 'sp_at_low_wb', 'wb_sp_scm'
        ]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features từ raw inputs"""
        df = df.copy()
        for col in ['fly_ash', 'slag', 'silica_fume']:
            if col not in df.columns:
                df[col] = 0

        df['binder'] = df['cement'] + df['fly_ash'] + df['slag'] + df['silica_fume']
        df['w_b'] = df['water'] / df['binder'].replace(0, np.nan)
        df['scm_frac'] = (df['fly_ash'] + df['slag'] + df['silica_fume']) / df['binder'].replace(0, np.nan)
        df['agg_total'] = df['fine_agg'] + df['coarse_agg']
        df['sand_ratio'] = df['fine_agg'] / df['agg_total'].replace(0, np.nan)

        df['sp_per_b'] = df['sp'] / df['binder'].replace(0, np.nan)
        df['sp_per_w'] = df['sp'] / df['water'].replace(0, np.nan)

        df['paste_volume'] = (df['binder'] / 3150) + (df['water'] / 1000)
        df['paste_to_agg'] = df['paste_volume'] / df['agg_total'].replace(0, np.nan)

        df['effective_w_c'] = df['water'] / df['cement'].replace(0, np.nan)
        df['pozzolanic_idx'] = (df['fly_ash'] + df['slag'] * 1.2 + df['silica_fume'] * 2.0) / df['binder'].replace(0, np.nan)

        df['w_b_x_scm'] = df['w_b'] * df['scm_frac']
        df['w_b_x_sp'] = df['w_b'] * df['sp_per_b']
        df['sp_x_scm'] = df['sp_per_b'] * df['scm_frac']

        df['w_b_sq'] = df['w_b'] ** 2
        df['sp_per_b_sq'] = df['sp_per_b'] ** 2

        df['log_sp'] = np.log1p(df['sp'])
        df['log_silica_fume'] = np.log1p(df['silica_fume'])

        df['sp_saturation'] = 1 - np.exp(-df['sp_per_b'] * 100)
        df['excess_water_idx'] = (df['w_b'] - 0.42).clip(lower=0)
        df['sp_at_low_wb'] = df['sp_per_b'] * np.exp(-df['w_b'] / 0.3)
        df['wb_sp_scm'] = df['w_b'] * df['sp_per_b'] * df['scm_frac']

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df[self.feature_names] = df[self.feature_names].fillna(0)

        return df[self.feature_names]

if 'OptimalSlumpFeatureBuilder' not in sys.modules['__main__'].__dict__:
    sys.modules['__main__'].OptimalSlumpFeatureBuilder = OptimalSlumpFeatureBuilder

# =========================================================================
# BƯỚC 2: IMPORT CONFIG (Đường dẫn models - LOCAL VERSION)
# =========================================================================
try:
    from config import F28_MODEL, S_MODEL, SLUMP_MODEL, MODELS_DIR
    F28_MODEL_PATH = F28_MODEL
    S_MODEL_PATH = S_MODEL
    SLUMP_MODEL_PATH = SLUMP_MODEL
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = BASE_DIR / "models"
    F28_MODEL_PATH = MODELS_DIR / "f28_blend_retrained_bundle.pkl"
    S_MODEL_PATH = MODELS_DIR / "s_model_catboost_mono_s_bundle.pkl"
    SLUMP_MODEL_PATH = MODELS_DIR / "feature_builder_final.pkl"
    print(f"⚠️ Config import failed, using fallback paths:")
    print(f"   MODELS_DIR: {MODELS_DIR}")

# =========================================================================
# BƯỚC 3: UNIFIED PREDICTOR - LOCAL VERSION
# =========================================================================
class UnifiedPredictor:
    """
    ✅ LOCAL VERSION - LOAD MODELS TỪ THỦ MỤC models/
    - Tự động tìm sub-bundles (CatBoost, XGBoost) trong cùng folder
    - Feature engineering đúng với training
    - Tương thích với NSGA-II optimization
    """

    def __init__(self):
        """Load all models from disk"""
        self.f28_bundle = None
        self.s_bundle = None
        self.slump_builder = None
        self.slump_models = {}
        self.cat_bundle = None
        self.xgb_bundle = None
        self._load_models()

    def _load_models(self):
        """Load models with error handling - LOCAL VERSION"""
        print("\n⏳ Đang tải các mô hình AI (LOCAL)...")
        print(f"📂 Models directory: {MODELS_DIR}")
        print(f"📂 MODELS_DIR exists: {MODELS_DIR.exists()}")
        print(f"📂 Current working directory: {Path.cwd()}")

        if F28_MODEL_PATH.exists():
            try:
                print(f"   Loading: {F28_MODEL_PATH.name}")
                self.f28_bundle = joblib.load(F28_MODEL_PATH)
                
                models_dir = F28_MODEL_PATH.parent
                
                cat_bundle_path = models_dir / "f28_catboost_retrained_bundle.pkl"
                if cat_bundle_path.exists():
                    print(f"   → Loading CatBoost: {cat_bundle_path.name}")
                    self.cat_bundle = joblib.load(cat_bundle_path)
                else:
                    print(f"   ⚠️ CatBoost sub-bundle not found: {cat_bundle_path.name}")
                
                xgb_bundle_path = models_dir / "f28_xgboost_retrained_bundle.pkl"
                if xgb_bundle_path.exists():
                    print(f"   → Loading XGBoost: {xgb_bundle_path.name}")
                    self.xgb_bundle = joblib.load(xgb_bundle_path)
                else:
                    print(f"   ⚠️ XGBoost sub-bundle not found: {xgb_bundle_path.name}")

                print(f"✅ F28 Blend model loaded successfully!")
                
            except Exception as e:
                print(f"❌ Error loading F28: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"❌ F28 model not found: {F28_MODEL_PATH}")

        if S_MODEL_PATH.exists():
            try:
                print(f"   Loading: {S_MODEL_PATH.name}")
                self.s_bundle = joblib.load(S_MODEL_PATH)
                print(f"✅ S model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading S: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"❌ S model not found: {S_MODEL_PATH}")

        print(f"\n🔍 DEBUG SLUMP MODEL:")
        print(f"   SLUMP_MODEL_PATH: {SLUMP_MODEL_PATH}")
        print(f"   Exists: {SLUMP_MODEL_PATH.exists()}")
        
        if SLUMP_MODEL_PATH.exists():
            try:
                print(f"   Loading: {SLUMP_MODEL_PATH.name}")
                self.slump_builder = joblib.load(SLUMP_MODEL_PATH)
                print(f"   ✅ Slump builder loaded: {type(self.slump_builder)}")

                parent_dir = SLUMP_MODEL_PATH.parent
                
                slump_models_dir_1 = parent_dir / "slump_models"
                print(f"   Checking: {slump_models_dir_1}")
                print(f"   Exists: {slump_models_dir_1.exists()}")
                
                slump_models_dir_2 = parent_dir / "models"
                print(f"   Checking: {slump_models_dir_2}")
                print(f"   Exists: {slump_models_dir_2.exists()}")
                
                if slump_models_dir_1.exists():
                    slump_models_dir = slump_models_dir_1
                elif slump_models_dir_2.exists():
                    slump_models_dir = slump_models_dir_2
                else:
                    print(f"   ❌ No slump_models directory found!")
                    slump_models_dir = None
                
                if slump_models_dir:
                    print(f"   → Using: {slump_models_dir}")
                    
                    all_pkl_files = list(slump_models_dir.glob("*.pkl"))
                    print(f"   → Found {len(all_pkl_files)} .pkl files:")
                    for f in all_pkl_files[:5]:
                        print(f"      - {f.name}")
                    
                    loaded_folds = 0
                    for fold in range(1, 11):
                        lgbm_path = slump_models_dir / f"slump_lgbm_fold_{fold}.pkl"
                        xgb_path = slump_models_dir / f"slump_xgboost_fold_{fold}.pkl"

                        if lgbm_path.exists() and xgb_path.exists():
                            try:
                                self.slump_models[f'fold_{fold}'] = {
                                    'lgbm': joblib.load(lgbm_path),
                                    'xgboost': joblib.load(xgb_path)
                                }
                                loaded_folds += 1
                            except Exception as e:
                                print(f"      ❌ Error loading fold {fold}: {e}")
                    
                    if loaded_folds > 0:
                        print(f"✅ Slump model loaded ({loaded_folds}/10 folds)")
                    else:
                        print(f"❌ No slump sub-models loaded (0/10 folds)")
                        
            except Exception as e:
                print(f"❌ Error loading Slump: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"❌ Slump builder not found: {SLUMP_MODEL_PATH}")

        print("\n" + "="*60)

    def _build_features(self, mix_design: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Build features cho cả 3 models"""
        defaults = {
            'cement': 0, 'water': 0, 'flyash': 0, 'slag': 0,
            'silica_fume': 0, 'superplasticizer': 0,
            'fine_agg': 0, 'coarse_agg': 0,
            'accelerating_agent': 0
        }
        mix = {k: mix_design.get(k, defaults[k]) for k in defaults}

        binder = mix['cement'] + mix['flyash'] + mix['slag'] + mix['silica_fume']
        safe_binder = binder if binder > 0 else 1.0

        w_b = mix['water'] / safe_binder
        scm = mix['flyash'] + mix['slag'] + mix['silica_fume']
        scm_frac = scm / safe_binder

        total_agg = mix['fine_agg'] + mix['coarse_agg']
        safe_agg = total_agg if total_agg > 0 else 1.0
        sand_ratio = mix['fine_agg'] / safe_agg

        w_b_sq = w_b ** 2
        w_b_cube = w_b ** 3

        w_b_x_scm = w_b * scm_frac
        cement_x_sf = mix['cement'] * mix['silica_fume']
        w_b_x_binder = w_b * binder

        cement_frac = mix['cement'] / safe_binder
        sf_frac = mix['silica_fume'] / safe_binder
        paste_volume = (binder + mix['water']) / 2.5

        log_binder = np.log1p(binder)
        log_w_b = np.log(np.clip(w_b, 0.2, 1.0))

        f28_s_features = np.array([[
            mix['cement'], mix['water'], mix['flyash'], mix['slag'], mix['silica_fume'],
            mix['superplasticizer'], mix['fine_agg'], mix['coarse_agg'], mix['accelerating_agent'],
            binder, w_b, scm, scm_frac, sand_ratio,
            w_b_sq, w_b_cube, w_b_x_scm, cement_x_sf, w_b_x_binder,
            cement_frac, sf_frac, total_agg, paste_volume,
            log_binder, log_w_b
        ]], dtype=float)

        sp_per_b = mix['superplasticizer'] / safe_binder
        sp_per_w = mix['superplasticizer'] / mix['water'] if mix['water'] > 0 else 0
        paste_to_agg = paste_volume / safe_agg
        effective_w_c = mix['water'] / mix['cement'] if mix['cement'] > 0 else 0.5
        pozzolanic_idx = (mix['flyash'] + mix['slag']*1.2 + mix['silica_fume']*2.0) / safe_binder

        w_b_x_sp = w_b * sp_per_b
        sp_x_scm = sp_per_b * scm_frac
        sp_per_b_sq = sp_per_b ** 2
        log_sp = np.log1p(mix['superplasticizer'])
        log_silica_fume = np.log1p(mix['silica_fume'])

        sp_saturation = 1 - np.exp(-sp_per_b * 100)
        excess_water_idx = np.maximum(w_b - 0.42, 0)
        sp_at_low_wb = sp_per_b * np.exp(-w_b / 0.3)
        wb_sp_scm = w_b * sp_per_b * scm_frac

        slump_features = np.array([[
            mix['cement'], mix['water'], mix['fine_agg'], mix['coarse_agg'], mix['superplasticizer'],
            mix['flyash'], mix['slag'], mix['silica_fume'],
            binder, w_b, scm_frac, sand_ratio, sp_per_b, sp_per_w,
            paste_volume, total_agg, paste_to_agg, effective_w_c, pozzolanic_idx,
            w_b_x_scm, w_b_x_sp, sp_x_scm, w_b_sq, sp_per_b_sq,
            log_sp, log_silica_fume,
            sp_saturation, excess_water_idx, sp_at_low_wb, wb_sp_scm
        ]], dtype=float)

        return {
            'f28': f28_s_features,
            's': f28_s_features,
            'slump': slump_features
        }

    def predict_f28(self, mix_design: Dict[str, float]) -> float:
        """Predict f28 (MPa) - REAL AI"""
        if not self.f28_bundle:
            print("⚠️ F28 model not loaded")
            return 0.0

        try:
            features = self._build_features(mix_design)
            X = features['f28']

            w_cat = self.f28_bundle.get('w_cat', 0.5)
            w_xgb = self.f28_bundle.get('w_xgb', 0.5)
            blend_iso = self.f28_bundle.get('blend_iso')

            cat_preds = []
            if self.cat_bundle and 'models' in self.cat_bundle:
                for model in self.cat_bundle['models']:
                    cat_preds.append(model.predict(X)[0])
                cat_mean = np.mean(cat_preds)
                cat_iso = self.cat_bundle.get('iso')
                cat_final = cat_iso.transform([cat_mean])[0] if cat_iso else cat_mean
            else:
                cat_final = 0

            xgb_preds = []
            if self.xgb_bundle and 'models' in self.xgb_bundle:
                for model in self.xgb_bundle['models']:
                    xgb_preds.append(model.predict(X)[0])
                xgb_mean = np.mean(xgb_preds)
                xgb_iso = self.xgb_bundle.get('iso')
                xgb_final = xgb_iso.transform([xgb_mean])[0] if xgb_iso else xgb_mean
            else:
                xgb_final = 0

            blend = w_cat * cat_final + w_xgb * xgb_final
            final = blend_iso.transform([blend])[0] if blend_iso else blend

            return float(np.clip(final, 5, 120))

        except Exception as e:
            print(f"❌ Error predicting f28: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def predict_s(self, mix_design: Dict[str, float]) -> float:
        """Predict s parameter - REAL AI"""
        if not self.s_bundle:
            return 0.25

        try:
            features = self._build_features(mix_design)
            X = features['s']

            models = self.s_bundle.get('models', [])
            iso = self.s_bundle.get('iso')

            if not models:
                return 0.25

            preds = [model.predict(X)[0] for model in models]
            pred_mean = np.mean(preds)
            pred_iso = iso.transform([pred_mean])[0] if iso else pred_mean

            return float(np.clip(pred_iso, 0.12, 0.60))

        except Exception as e:
            print(f"❌ Error predicting s: {e}")
            return 0.25

    def predict_slump(self, mix_design: Dict[str, float]) -> float:
        """Predict slump (mm) - REAL AI"""
        if not self.slump_models:
            print("⚠️ Slump models not loaded")
            return 0.0

        try:
            features = self._build_features(mix_design)
            X = features['slump']

            feature_names = [
                'cement', 'water', 'fine_agg', 'coarse_agg', 'sp',
                'fly_ash', 'slag', 'silica_fume',
                'binder', 'w_b', 'scm_frac', 'sand_ratio', 'sp_per_b', 'sp_per_w',
                'paste_volume', 'agg_total', 'paste_to_agg', 'effective_w_c', 'pozzolanic_idx',
                'w_b_x_scm', 'w_b_x_sp', 'sp_x_scm', 'w_b_sq', 'sp_per_b_sq',
                'log_sp', 'log_silica_fume',
                'sp_saturation', 'excess_water_idx', 'sp_at_low_wb', 'wb_sp_scm'
            ]

            predictions = []
            for fold_models in self.slump_models.values():
                pred_lgbm = fold_models['lgbm'].predict(X)[0]

                import xgboost as xgb
                dtest = xgb.DMatrix(X, feature_names=feature_names)
                pred_xgb = fold_models['xgboost'].predict(dtest)[0]

                pred_ensemble = (pred_lgbm + pred_xgb) / 2
                predictions.append(pred_ensemble)

            pred_mean = np.mean(predictions)
            return float(np.clip(pred_mean, 0, 300))

        except Exception as e:
            print(f"❌ Error predicting slump: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def predict_strength_at_age(self, mix_design: Dict[str, float], age: int) -> float:
        """Predict cường độ tại tuổi bất kỳ"""
        if age == 28:
            return self.predict_f28(mix_design)

        f28 = self.predict_f28(mix_design)
        s = self.predict_s(mix_design)

        beta = np.exp(s * (1.0 - np.sqrt(28.0 / age)))
        f_age = f28 * beta

        return float(f_age)

    # =====================================================================
    # BATCH API (nhanh): dự đoán cho cả quần thể trong 1 lần gọi
    # Kết quả khớp với predict_f28 / predict_s / predict_slump (từng dòng)
    # =====================================================================
    MIX_KEYS = ['cement', 'water', 'flyash', 'slag', 'silica_fume',
                'superplasticizer', 'fine_agg', 'coarse_agg']

    SLUMP_FEATURE_NAMES = [
        'cement', 'water', 'fine_agg', 'coarse_agg', 'sp',
        'fly_ash', 'slag', 'silica_fume',
        'binder', 'w_b', 'scm_frac', 'sand_ratio', 'sp_per_b', 'sp_per_w',
        'paste_volume', 'agg_total', 'paste_to_agg', 'effective_w_c', 'pozzolanic_idx',
        'w_b_x_scm', 'w_b_x_sp', 'sp_x_scm', 'w_b_sq', 'sp_per_b_sq',
        'log_sp', 'log_silica_fume',
        'sp_saturation', 'excess_water_idx', 'sp_at_low_wb', 'wb_sp_scm'
    ]

    def _build_features_batch(self, M: np.ndarray):
        """
        Vector hóa _build_features.
        M: (n, 8) theo thứ tự MIX_KEYS. Trả về (X_f28_s [n,25], X_slump [n,30]).
        Công thức giữ NGUYÊN như _build_features (kể cả paste_volume).
        """
        M = np.asarray(M, dtype=float)
        cement, water, flyash, slag, sf, sp, fine, coarse = (M[:, i] for i in range(8))
        acc = np.zeros_like(cement)  # accelerating_agent = 0

        binder = cement + flyash + slag + sf
        safe_binder = np.where(binder > 0, binder, 1.0)
        w_b = water / safe_binder
        scm = flyash + slag + sf
        scm_frac = scm / safe_binder

        total_agg = fine + coarse
        safe_agg = np.where(total_agg > 0, total_agg, 1.0)
        sand_ratio = fine / safe_agg

        w_b_sq = w_b ** 2
        w_b_cube = w_b ** 3
        w_b_x_scm = w_b * scm_frac
        cement_x_sf = cement * sf
        w_b_x_binder = w_b * binder
        cement_frac = cement / safe_binder
        sf_frac = sf / safe_binder
        paste_volume = (binder + water) / 2.5
        log_binder = np.log1p(binder)
        log_w_b = np.log(np.clip(w_b, 0.2, 1.0))

        X_f28_s = np.column_stack([
            cement, water, flyash, slag, sf, sp, fine, coarse, acc,
            binder, w_b, scm, scm_frac, sand_ratio,
            w_b_sq, w_b_cube, w_b_x_scm, cement_x_sf, w_b_x_binder,
            cement_frac, sf_frac, total_agg, paste_volume,
            log_binder, log_w_b
        ])

        sp_per_b = sp / safe_binder
        sp_per_w = np.where(water > 0, sp / np.where(water > 0, water, 1.0), 0.0)
        paste_to_agg = paste_volume / safe_agg
        effective_w_c = np.where(cement > 0, water / np.where(cement > 0, cement, 1.0), 0.5)
        pozzolanic_idx = (flyash + slag * 1.2 + sf * 2.0) / safe_binder

        w_b_x_sp = w_b * sp_per_b
        sp_x_scm = sp_per_b * scm_frac
        sp_per_b_sq = sp_per_b ** 2
        log_sp = np.log1p(sp)
        log_silica_fume = np.log1p(sf)

        sp_saturation = 1 - np.exp(-sp_per_b * 100)
        excess_water_idx = np.maximum(w_b - 0.42, 0)
        sp_at_low_wb = sp_per_b * np.exp(-w_b / 0.3)
        wb_sp_scm = w_b * sp_per_b * scm_frac

        X_slump = np.column_stack([
            cement, water, fine, coarse, sp,
            flyash, slag, sf,
            binder, w_b, scm_frac, sand_ratio, sp_per_b, sp_per_w,
            paste_volume, total_agg, paste_to_agg, effective_w_c, pozzolanic_idx,
            w_b_x_scm, w_b_x_sp, sp_x_scm, w_b_sq, sp_per_b_sq,
            log_sp, log_silica_fume,
            sp_saturation, excess_water_idx, sp_at_low_wb, wb_sp_scm
        ])
        return X_f28_s, X_slump

    @staticmethod
    def _mean_models(models, X) -> np.ndarray:
        """Mean dự đoán của list model, mỗi model gọi predict 1 lần cho cả lô"""
        return np.mean([np.asarray(m.predict(X), dtype=float).reshape(-1) for m in models], axis=0)

    def _predict_f28_X(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if not self.f28_bundle:
            return np.zeros(n)

        w_cat = self.f28_bundle.get('w_cat', 0.5)
        w_xgb = self.f28_bundle.get('w_xgb', 0.5)
        blend_iso = self.f28_bundle.get('blend_iso')

        if self.cat_bundle and 'models' in self.cat_bundle:
            cat_mean = self._mean_models(self.cat_bundle['models'], X)
            cat_iso = self.cat_bundle.get('iso')
            cat_final = cat_iso.transform(cat_mean) if cat_iso else cat_mean
        else:
            cat_final = np.zeros(n)

        if self.xgb_bundle and 'models' in self.xgb_bundle:
            xgb_mean = self._mean_models(self.xgb_bundle['models'], X)
            xgb_iso = self.xgb_bundle.get('iso')
            xgb_final = xgb_iso.transform(xgb_mean) if xgb_iso else xgb_mean
        else:
            xgb_final = np.zeros(n)

        blend = w_cat * cat_final + w_xgb * xgb_final
        final = blend_iso.transform(blend) if blend_iso else blend
        return np.clip(np.asarray(final, dtype=float), 5, 120)

    def _predict_s_X(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if not self.s_bundle:
            return np.full(n, 0.25)
        models = self.s_bundle.get('models', [])
        if not models:
            return np.full(n, 0.25)
        iso = self.s_bundle.get('iso')
        mean = self._mean_models(models, X)
        out = iso.transform(mean) if iso else mean
        return np.clip(np.asarray(out, dtype=float), 0.12, 0.60)

    def _predict_slump_X(self, X: np.ndarray, max_folds: Optional[int] = None) -> np.ndarray:
        n = X.shape[0]
        if not self.slump_models:
            return np.zeros(n)
        import xgboost as xgb
        dtest = xgb.DMatrix(X, feature_names=self.SLUMP_FEATURE_NAMES)  # tạo 1 lần, dùng cho mọi fold
        folds = list(self.slump_models.values())
        if max_folds:
            folds = folds[:int(max_folds)]
        preds = []
        for fm in folds:
            p_lgbm = np.asarray(fm['lgbm'].predict(X), dtype=float).reshape(-1)
            p_xgb = np.asarray(fm['xgboost'].predict(dtest), dtype=float).reshape(-1)
            preds.append((p_lgbm + p_xgb) / 2)
        return np.clip(np.mean(preds, axis=0), 0, 300)

    def predict_batch(self, M, max_slump_folds: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Dự đoán f28, s, slump cho cả lô.
        M: array (n, 8) theo thứ tự MIX_KEYS, hoặc list[dict] mix design.
        max_slump_folds: chỉ dùng k fold đầu cho slump (None = dùng hết, kết quả y hệt predict_all).
        """
        if isinstance(M, (list, tuple)) and len(M) > 0 and isinstance(M[0], dict):
            M = np.array([[d.get(k, 0.0) for k in self.MIX_KEYS] for d in M], dtype=float)
        M = np.atleast_2d(np.asarray(M, dtype=float))
        X_f28_s, X_slump = self._build_features_batch(M)
        return {
            'f28': self._predict_f28_X(X_f28_s),
            's': self._predict_s_X(X_f28_s),
            'slump': self._predict_slump_X(X_slump, max_slump_folds),
        }

    def predict_all(self, mix_design: Dict[str, float]) -> Dict[str, float]:
        """Predict tất cả properties - INTERFACE CHO NSGA-II"""
        return {
            'f28': self.predict_f28(mix_design),
            's': self.predict_s(mix_design),
            'slump': self.predict_slump(mix_design)
        }


if __name__ == "__main__":
    print("="*60)
    print("TESTING UNIFIED PREDICTOR (LOCAL VERSION)")
    print("="*60)
    
    predictor = UnifiedPredictor()

    test_mix = {
        'cement': 350, 'water': 160, 'flyash': 50, 'slag': 80,
        'silica_fume': 20, 'superplasticizer': 6.5,
        'fine_agg': 750, 'coarse_agg': 1050
    }

    print("\n" + "="*60)
    print("TEST MIX DESIGN:")
    print("="*60)
    for key, value in test_mix.items():
        print(f"  {key:<20} {value:>8.1f}")

    print("\n" + "="*60)
    print("PREDICTIONS:")
    print("="*60)

    predictions = predictor.predict_all(test_mix)
    print(f"f28:   {predictions['f28']:.1f} MPa")
    print(f"s:     {predictions['s']:.3f}")
    print(f"Slump: {predictions['slump']:.0f} mm")

    print("\n" + "="*60)
    if predictions['f28'] > 0 and predictions['slump'] > 0:
        print("✅ REAL AI PREDICTOR READY FOR OPTIMIZATION!")
    else:
        print("⚠️ Models chưa load đúng, check models/ folder!")
    print("="*60)