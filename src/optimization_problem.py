"""
optimization_problem.py - Định nghĩa optimization problem cho pymoo
✅ FIXED: Strength objective giờ penalize khi vượt quá 120% target
✅ FIXED: Thêm __deepcopy__ để tránh lỗi XGBoost bad allocation
"""
import numpy as np
import copy
from pymoo.core.problem import Problem
from typing import Dict, Tuple, List

# SAFE IMPORT
try:
    from predictor_unified import UnifiedPredictor
    from material_database import MaterialDatabase
    from cost_calculator import CostCalculator
    from co2_calculator import CO2Calculator
    from cement_adjuster import CementTypeAdjuster
except ImportError:
    from src.predictor_unified import UnifiedPredictor
    from src.material_database import MaterialDatabase
    from src.cost_calculator import CostCalculator
    from src.co2_calculator import CO2Calculator
    from src.cement_adjuster import CementTypeAdjuster


class ConcreteMixOptimizationProblem(Problem):
    """
    Multi-objective optimization problem cho concrete mix design
    ✅ FIXED: Strength objective với penalty khi vượt 120% target
    """

    def __init__(
        self,
        predictor: UnifiedPredictor,
        constraint_config: Dict,
        cement_type: str = 'PC40'
    ):
        self.predictor = predictor
        self.config = constraint_config
        self.cement_type = cement_type

        # ✅ Lấy fc_target để tính penalty
        self.fc_target = constraint_config['user_input'].get('fc_target', 40)
        self.fc_upper_limit = self.fc_target * 1.20  # 120% target

        # Initialize calculators
        self.material_db = MaterialDatabase()
        self.cost_calc = CostCalculator(self.material_db)
        self.co2_calc = CO2Calculator(self.material_db)
        self.adjuster = CementTypeAdjuster()

        # Extract bounds
        bounds_dict = constraint_config['bounds']
        self.var_names = ['cement', 'water', 'flyash', 'slag', 'silica_fume',
                          'superplasticizer', 'fine_agg', 'coarse_agg']

        xl = np.array([bounds_dict[k][0] for k in self.var_names])
        xu = np.array([bounds_dict[k][1] for k in self.var_names])

        # Number of objectives and constraints
        n_var = 8
        n_obj = 4  # cost, strength, slump, co2
        n_constr = self._count_constraints(constraint_config)

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu
        )

        self.use_vectorized = True
        self._prepare_vectorized()

    def __deepcopy__(self, memo):
        """Custom deepcopy để tránh copy predictor (chứa XGBoost models)"""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        for k, v in self.__dict__.items():
            if k == 'predictor':
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        
        return result

    def _count_constraints(self, config: Dict) -> int:
        """Đếm số constraints"""
        count = 0
        for c in config['constraints']:
            if c['type'] in ['w_b_range', 'binder_range']:
                count += 2
            else:
                count += 1
        return count

    def _x_to_mix_dict(self, x: np.ndarray) -> Dict[str, float]:
        """Convert decision vector to mix design dict"""
        return {name: float(x[i]) for i, name in enumerate(self.var_names)}
    
    def validate_mix(self, mix: Dict) -> Tuple[bool, List[str]]:
        """Validate mix design dựa trên constraints hiện tại"""
        violations = []
        
        bounds = self.config['bounds']
        for key, val in mix.items():
            if key in bounds:
                min_v, max_v = bounds[key]
                if not (min_v <= val <= max_v):
                    violations.append(f"{key}={val:.1f} ngoài [{min_v}, {max_v}]")
        
        vol = self.material_db.compute_volume(mix)
        if abs(vol - 1.0) > 0.02:
            violations.append(f"Volume={vol:.3f} (target 1.0)")
            
        binder = sum(mix.get(k, 0) for k in ['cement', 'flyash', 'slag', 'silica_fume'])
        if binder > 0:
            w_b = mix.get('water', 0) / binder
            if not (0.25 <= w_b <= 0.65):
                violations.append(f"w/b={w_b:.2f} ngoài [0.25, 0.65]")

        return (len(violations) == 0, violations)

    def _calculate_strength_objective(self, fc_age: float) -> float:
        """
        ✅ NEW: Tính strength objective với penalty khi vượt 120% target
        
        Logic:
        - fc < fc_target: Penalty lớn (không đạt yêu cầu)
        - fc_target ≤ fc ≤ 1.20×fc_target: Maximize (minimize negative)
        - fc > 1.20×fc_target: Penalty nhẹ (quá mạnh = lãng phí)
        """
        if fc_age < self.fc_target:
            # Không đạt target → penalty lớn
            penalty = (self.fc_target - fc_age) * 10
            return -self.fc_target + penalty
        
        elif fc_age <= self.fc_upper_limit:
            # Trong vùng tối ưu (100-120% target) → maximize
            return -fc_age
        
        else:
            # Vượt quá 120% → penalty nhẹ (discourage over-design)
            excess = fc_age - self.fc_upper_limit
            penalty = excess * 2  # Penalty nhẹ hơn so với under-design
            return -self.fc_upper_limit + penalty

    # =====================================================================
    # VECTOR HÓA: đánh giá cả quần thể trong 1 lần (kết quả khớp _evaluate_loop)
    # =====================================================================
    def _prepare_vectorized(self):
        """Tính sẵn đơn giá, hệ số CO2, tỷ trọng và hệ số hiệu chỉnh xi măng"""
        n = len(self.var_names)
        price = np.zeros(n)
        co2_unit = np.zeros(n)
        dens = np.zeros(n)
        for i, name in enumerate(self.var_names):
            if name == 'cement':
                props = self.material_db.get_cement_props(self.cement_type)
            else:
                props = self.material_db.get_material_by_mix_key(name)
            if props:
                price[i] = props['price']
                co2_unit[i] = (props['co2_production']
                               + props['transport_distance'] * props['co2_transport'])
            dens[i] = self.material_db.get_density(name)
        self._price = price
        self._co2_unit = co2_unit
        self._dens = dens
        self._a3 = self.co2_calc.a3_emission
        self._factors = self.adjuster.cement_factors.get(
            self.cement_type, self.adjuster.cement_factors['PC40'])

    def predict_batch(self, X):
        """
        Dự đoán + hiệu chỉnh xi măng cho cả lô.
        Trả về (f28, s, slump, fc_age) đều là mảng (n,).
        """
        M = np.atleast_2d(np.asarray(X, dtype=float))
        pred = self.predictor.predict_batch(M)
        fac = self._factors
        f28 = pred['f28'] * fac['f28']
        s = pred['s'] * fac['s_curve']
        slump = pred['slump'] / fac['water_demand']

        age_target = self.config['user_input']['age_target']
        if age_target == 28:
            fc_age = f28
        else:
            beta = np.exp(pred['s'] * (1.0 - np.sqrt(28.0 / age_target)))
            fc_age = pred['f28'] * beta * fac['f28']
        return f28, s, slump, fc_age

    def _strength_objective_vec(self, fc: np.ndarray) -> np.ndarray:
        under = -self.fc_target + (self.fc_target - fc) * 10
        mid = -fc
        over = -self.fc_upper_limit + (fc - self.fc_upper_limit) * 2
        return np.where(fc < self.fc_target, under,
                        np.where(fc <= self.fc_upper_limit, mid, over))

    def _evaluate_vectorized(self, X, out):
        M = np.atleast_2d(np.asarray(X, dtype=float))
        n = M.shape[0]
        ui = self.config['user_input']

        _, _, slump, fc_age = self.predict_batch(M)

        F = np.zeros((n, self.n_obj))
        G = np.zeros((n, self.n_constr))

        F[:, 0] = M @ self._price
        F[:, 1] = self._strength_objective_vec(fc_age)
        F[:, 2] = np.abs(slump - ui['slump_target'])
        F[:, 3] = M @ self._co2_unit + self._a3

        idx = {k: i for i, k in enumerate(self.var_names)}
        binder = M[:, idx['cement']] + M[:, idx['flyash']] + M[:, idx['slag']] + M[:, idx['silica_fume']]
        scm = M[:, idx['flyash']] + M[:, idx['slag']] + M[:, idx['silica_fume']]
        pos = binder > 0
        safe_b = np.where(pos, binder, 1.0)
        w_b = np.where(pos, M[:, idx['water']] / safe_b, 0.5)
        scm_frac = np.where(pos, scm / safe_b, 0.0)
        volume = np.where(M > 0, M / self._dens, 0.0).sum(axis=1)

        g = 0
        for c in self.config['constraints']:
            t = c['type']
            if t == 'slump_tolerance':
                G[:, g] = F[:, 2] - c['tolerance']; g += 1
            elif t == 'strength_min':
                G[:, g] = c['min_value'] - fc_age; g += 1
            elif t == 'w_b_range':
                G[:, g] = c['min'] - w_b; g += 1
                G[:, g] = w_b - c['max']; g += 1
            elif t == 'scm_limits':
                G[:, g] = scm_frac - c['total']; g += 1
            elif t == 'volume':
                G[:, g] = np.abs(volume - c['target']) - c['tolerance']; g += 1
            elif t == 'binder_range':
                G[:, g] = c['min'] - binder; g += 1
                G[:, g] = binder - c['max']; g += 1

        out["F"] = F
        out["G"] = G

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate population (ưu tiên bản vector hóa, lỗi thì rơi về bản từng dòng)"""
        if self.use_vectorized:
            try:
                self._evaluate_vectorized(X, out)
                return
            except Exception as e:
                print(f"⚠️ Vectorized evaluate lỗi ({e}); chuyển sang bản từng dòng.")
        self._evaluate_loop(X, out, *args, **kwargs)

    def _evaluate_loop(self, X, out, *args, **kwargs):
        """Evaluate population (bản gốc, từng cá thể)"""
        pop_size = X.shape[0]
        F = np.zeros((pop_size, self.n_obj))
        G = np.zeros((pop_size, self.n_constr))

        for i in range(pop_size):
            mix = self._x_to_mix_dict(X[i])

            # ===== PREDICT PROPERTIES =====
            try:
                predictions = self.predictor.predict_all(mix)
                predictions = self.adjuster.adjust_predictions(
                    mix, predictions, self.cement_type
                )

                # Predict strength at target age
                age_target = self.config['user_input']['age_target']
                if age_target == 28:
                    fc_age = predictions['f28']
                else:
                    fc_age = self.predictor.predict_strength_at_age(mix, age_target)
                    fc_age = self.adjuster.adjust_predictions(
                        mix, {'f28': fc_age}, self.cement_type
                    )['f28']

            except Exception as e:
                F[i] = [1e9, -0.1, 1000, 1e9]
                G[i] = np.ones(self.n_constr) * 1000
                continue

            # ===== CALCULATE COST & CO2 =====
            cost_data = self.cost_calc.calculate_total_cost(mix, self.cement_type)
            co2_data = self.co2_calc.calculate_total_emission(mix, self.cement_type)

            # ===== OBJECTIVES =====
            F[i, 0] = cost_data['total_cost']  # Minimize cost
            
            # ✅ FIXED: Dùng strength objective mới
            F[i, 1] = self._calculate_strength_objective(fc_age)
            
            F[i, 2] = abs(predictions['slump'] - self.config['user_input']['slump_target'])
            F[i, 3] = co2_data['total']  # Minimize CO2

            # ===== CONSTRAINTS (g <= 0) =====
            g_idx = 0
            binder = sum([
                mix.get('cement', 0),
                mix.get('flyash', 0),
                mix.get('slag', 0),
                mix.get('silica_fume', 0)
            ])
            scm = sum([
                mix.get('flyash', 0),
                mix.get('slag', 0),
                mix.get('silica_fume', 0)
            ])
            w_b = mix['water'] / binder if binder > 0 else 0.5
            volume = self.material_db.compute_volume(mix)

            for constraint in self.config['constraints']:
                if constraint['type'] == 'slump_tolerance':
                    G[i, g_idx] = F[i, 2] - constraint['tolerance']
                    g_idx += 1
                elif constraint['type'] == 'strength_min':
                    G[i, g_idx] = constraint['min_value'] - fc_age
                    g_idx += 1
                elif constraint['type'] == 'w_b_range':
                    G[i, g_idx] = constraint['min'] - w_b
                    g_idx += 1
                    G[i, g_idx] = w_b - constraint['max']
                    g_idx += 1
                elif constraint['type'] == 'scm_limits':
                    scm_frac = scm / binder if binder > 0 else 0
                    G[i, g_idx] = scm_frac - constraint['total']
                    g_idx += 1
                elif constraint['type'] == 'volume':
                    G[i, g_idx] = abs(volume - constraint['target']) - constraint['tolerance']
                    g_idx += 1
                elif constraint['type'] == 'binder_range':
                    G[i, g_idx] = constraint['min'] - binder
                    g_idx += 1
                    G[i, g_idx] = binder - constraint['max']
                    g_idx += 1

        out["F"] = F
        out["G"] = G


# ===== TEST =====
if __name__ == "__main__":
    print("✅ optimization_problem.py with smart strength penalty (100-120% target)")