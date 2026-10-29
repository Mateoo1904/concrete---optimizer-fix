"""
sensitivity_analyzer.py - Phân tích độ nhạy (sensitivity) & robustness cho mix tối ưu
✅ FIXED: Sử dụng predict_all() thay vì predict_properties()
✅ FIXED: Tương thích với CostCalculator và CO2Calculator API
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from datetime import datetime

# SAFE IMPORT (dùng giống các file khác)
try:
    from .predictor_unified import UnifiedPredictor
    from .material_database import MaterialDatabase
    from .cost_calculator import CostCalculator
    from .co2_calculator import CO2Calculator
except ImportError:  # chạy trực tiếp trong Colab (không phải package)
    from predictor_unified import UnifiedPredictor
    from material_database import MaterialDatabase
    from cost_calculator import CostCalculator
    from co2_calculator import CO2Calculator


class SensitivityAnalyzer:
    """
    Công cụ phân tích độ nhạy:
    - One-at-a-time (thay đổi từng biến)
    - Monte Carlo robustness (nhiễu ngẫu nhiên)
    """

    def __init__(
        self,
        predictor: Optional[UnifiedPredictor] = None,
        material_db: Optional[MaterialDatabase] = None
    ):
        self.predictor = predictor or UnifiedPredictor()
        self.material_db = material_db or MaterialDatabase()
        self.cost_calc = CostCalculator(self.material_db)
        self.co2_calc = CO2Calculator(self.material_db)

    def _predict_many(self, designs: List[Dict]) -> List[Dict]:
        """Dự đoán nhiều design cùng lúc (nhanh); rơi về predict_all nếu không có predict_batch"""
        if not designs:
            return []
        if hasattr(self.predictor, "predict_batch"):
            try:
                out = self.predictor.predict_batch(designs)
                return [
                    {"f28": float(out["f28"][i]), "s": float(out["s"][i]), "slump": float(out["slump"][i])}
                    for i in range(len(designs))
                ]
            except Exception as e:
                print(f"⚠️ predict_batch lỗi ({e}); dùng predict_all từng dòng.")
        return [self.predictor.predict_all(d) for d in designs]

    # ------------------------------------------------------------------
    # 1) One-at-a-time (OAT) – tăng/giảm từng thành phần
    # ------------------------------------------------------------------
    def one_at_a_time_analysis(
        self,
        base_design: Dict,
        step_pct: float = 0.05,
        cement_type: str = "PC40"
    ) -> pd.DataFrame:
        """
        Thay đổi từng thành phần ±step_pct, đo ảnh hưởng đến:
        - f28 (MPa)
        - slump (mm)
        - cost (VNĐ/m3)
        - CO2 (kg/m3)
        """
        factors = [
            "cement", "water", "flyash", "slag",
            "silica_fume", "superplasticizer",
            "fine_agg", "coarse_agg"
        ]

        rows = []

        # baseline - ✅ FIXED: Sử dụng predict_all()
        base_pred = self.predictor.predict_all(base_design)
        base_cost_data = self.cost_calc.calculate_total_cost(base_design, cement_type)
        base_co2_data = self.co2_calc.calculate_total_emission(base_design, cement_type)
        
        base_cost = base_cost_data['total_cost']
        base_co2 = base_co2_data['total']

        variants = []
        for factor in factors:
            if base_design.get(factor, 0) <= 0:
                continue

            for direction in ["-", "+"]:
                design = base_design.copy()
                delta = base_design[factor] * step_pct
                design[factor] = base_design[factor] + (delta if direction == "+" else -delta)
                variants.append((factor, direction, design))

        preds = self._predict_many([v[2] for v in variants])

        for (factor, direction, design), pred in zip(variants, preds):
            cost_data = self.cost_calc.calculate_total_cost(design, cement_type)
            co2_data = self.co2_calc.calculate_total_emission(design, cement_type)

            cost = cost_data['total_cost']
            co2 = co2_data['total']

            rows.append({
                "factor": factor,
                "direction": direction,
                "delta_pct": step_pct * (1 if direction == "+" else -1),
                "f28_baseline": base_pred["f28"],
                "f28_new": pred["f28"],
                "f28_change": pred["f28"] - base_pred["f28"],
                "slump_baseline": base_pred["slump"],
                "slump_new": pred["slump"],
                "slump_change": pred["slump"] - base_pred["slump"],
                "cost_baseline": base_cost,
                "cost_new": cost,
                "cost_change": cost - base_cost,
                "co2_baseline": base_co2,
                "co2_new": co2,
                "co2_change": co2 - base_co2
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 2) Monte Carlo robustness – nhiễu ngẫu nhiên các thành phần
    # ------------------------------------------------------------------
    def monte_carlo_robustness(
        self,
        base_design: Dict,
        n_samples: int = 200,
        variation_pct: float = 0.05,
        cement_type: str = "PC40",
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Monte Carlo: random nhiễu ±variation_pct cho mỗi thành phần
        để xem phân bố f28 / slump / cost / CO2
        """
        rng = np.random.default_rng(random_state)
        factors = [
            "cement", "water", "flyash", "slag",
            "silica_fume", "superplasticizer",
            "fine_agg", "coarse_agg"
        ]

        designs = []
        for _ in range(n_samples):
            design = base_design.copy()
            for f in factors:
                val = base_design.get(f, 0)
                noise = 1 + rng.uniform(-variation_pct, variation_pct)
                design[f] = max(val * noise, 0.0)
            designs.append(design)

        preds = self._predict_many(designs)

        rows = []
        for design, pred in zip(designs, preds):
            cost_data = self.cost_calc.calculate_total_cost(design, cement_type)
            co2_data = self.co2_calc.calculate_total_emission(design, cement_type)

            cost = cost_data['total_cost']
            co2 = co2_data['total']

            rows.append({
                **{f"{k}_kg": design.get(k, 0.0) for k in factors},
                "f28": pred["f28"],
                "slump": pred["slump"],
                "cost": cost,
                "co2": co2
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 3) Scenario analysis – một số kịch bản định sẵn
    # ------------------------------------------------------------------
    def scenario_analysis(
        self,
        base_design: Dict,
        cement_type: str = "PC40"
    ) -> pd.DataFrame:
        """
        Một số kịch bản:
        - Tăng SCM 10%
        - Giảm nước 5%
        - Tăng SP 20%
        """
        scenarios = {
            "Base": base_design,
            "SCM +10%": self._adjust_scm(base_design, 1.10),
            "Water -5%": self._scale_factor(base_design, "water", 0.95),
            "SP +20%": self._scale_factor(base_design, "superplasticizer", 1.20),
        }

        rows = []
        for name, mix in scenarios.items():
            # ✅ FIXED: Sử dụng predict_all()
            pred = self.predictor.predict_all(mix)
            cost_data = self.cost_calc.calculate_total_cost(mix, cement_type)
            co2_data = self.co2_calc.calculate_total_emission(mix, cement_type)
            
            cost = cost_data['total_cost']
            co2 = co2_data['total']
            
            rows.append({
                "scenario": name,
                "f28": pred["f28"],
                "slump": pred["slump"],
                "cost": cost,
                "co2": co2
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------
    def _scale_factor(self, design: Dict, factor: str, scale: float) -> Dict:
        d = design.copy()
        if factor in d:
            d[factor] = max(d[factor] * scale, 0.0)
        return d

    def _adjust_scm(self, design: Dict, scale: float) -> Dict:
        d = design.copy()
        for f in ["flyash", "slag", "silica_fume"]:
            if f in d:
                d[f] = max(d[f] * scale, 0.0)
        return d


if __name__ == "__main__":
    print("✅ SensitivityAnalyzer ready – dùng với mix tối ưu để phân tích độ nhạy.")