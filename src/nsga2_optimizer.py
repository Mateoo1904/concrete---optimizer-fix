"""
nsga2_optimizer.py - NSGA-II optimizer wrapper
✅ FIXED V3: Strength Optimized target 125-130% fc_target
✅ Đảm bảo Strength Optimized luôn mạnh nhất
"""
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from typing import Dict, List

# SAFE IMPORT
try:
    from optimization_problem import ConcreteMixOptimizationProblem
    from predictor_unified import UnifiedPredictor
    from constraint_builder import ConstraintBuilder
    from material_database import MaterialDatabase
except ImportError:
    from src.optimization_problem import ConcreteMixOptimizationProblem
    from src.predictor_unified import UnifiedPredictor
    from src.constraint_builder import ConstraintBuilder
    from src.material_database import MaterialDatabase


class MixDesignOptimizer:
    """
    NSGA-II optimizer cho concrete mix design
    """

    def __init__(
        self,
        predictor: UnifiedPredictor,
        material_db: MaterialDatabase,
        pop_size: int = 100,
        n_gen: int = 200,
        seed: int = 42,
        early_stop: bool = False
    ):
        self.early_stop = early_stop
        self.predictor = predictor
        self.material_db = material_db
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.seed = seed
        self.results = {}

    def optimize(
        self,
        user_input: Dict,
        cement_types: List[str] = ['PC40'],
        verbose: bool = True
    ) -> Dict:
        """
        Chạy optimization cho các loại xi măng
        """
        results_all = {}

        for cement_type in cement_types:
            if verbose:
                print(f"\n{'='*70}")
                print(f"🔄 Optimizing for {cement_type}")
                print(f"{'='*70}")

            result = self._optimize_single_cement(
                user_input, cement_type, verbose
            )
            results_all[cement_type] = result

        self.results = results_all
        return results_all

    def _optimize_single_cement(
        self,
        user_input: Dict,
        cement_type: str,
        verbose: bool
    ) -> Dict:
        """Optimize cho 1 loại xi măng"""

        # Build constraints
        builder = ConstraintBuilder(self.material_db)
        constraint_config = builder.build_from_user_input(user_input)

        if verbose:
            print(builder.get_constraint_summary())

        # Create problem
        problem = ConcreteMixOptimizationProblem(
            self.predictor,
            constraint_config,
            cement_type
        )

        # Setup algorithm
        algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=LatinHypercubeSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=0.2, eta=20),
            eliminate_duplicates=True
        )

        # Run optimization
        if verbose:
            print(f"\n🚀 Running NSGA-II (pop={self.pop_size}, gen={self.n_gen})...")
            print(f"   This may take 3-5 minutes...")

        if self.early_stop:
            # Dừng sớm khi Pareto front hội tụ (tối đa n_gen thế hệ)
            from pymoo.termination.default import DefaultMultiObjectiveTermination
            termination = DefaultMultiObjectiveTermination(
                xtol=1e-4, ftol=0.005, period=20,
                n_max_gen=self.n_gen, n_max_evals=int(1e9)
            )
        else:
            termination = ('n_gen', self.n_gen)

        res = minimize(
            problem,
            algorithm,
            termination=termination,
            seed=self.seed,
            verbose=verbose,
            save_history=False
        )

        # Extract results
        X_pareto = res.X
        F_pareto = res.F

        if verbose:
            print(f"\n✅ Optimization complete!")
            print(f"   Pareto front: {len(X_pareto)} solutions")

        # Select diverse designs
        top_designs = self._select_diverse_designs(
            X_pareto, F_pareto, problem, user_input, n=5
        )

        # Calculate metrics
        metrics = self._calculate_metrics(F_pareto)

        return {
            'pareto_front': (X_pareto, F_pareto),
            'top_designs': top_designs,
            'metrics': metrics,
            'problem': problem,
        }

    def _select_diverse_designs(
        self,
        X: np.ndarray,
        F: np.ndarray,
        problem: ConcreteMixOptimizationProblem,
        user_input: Dict,
        n: int = 5
    ) -> List[Dict]:
        """
        ✅ FIXED V3: Strength Optimized target 125-130% fc_target
        """
        designs = []
        fc_target = user_input.get('fc_target', 40)
        
        # ===== IMPORTANT: Tính actual strength cho mỗi design =====
        # Nhanh: dự đoán cả Pareto front trong 1 lần; lỗi thì rơi về vòng lặp từng dòng
        try:
            _, _, _, fc_age_all = problem.predict_batch(X)
            actual_strengths = np.asarray(fc_age_all, dtype=float)
        except Exception:
            actual_strengths = self._actual_strengths_loop(X, problem, user_input)

        # 1. Cost Optimized - Cheapest
        idx_cheap = np.argmin(F[:, 0])
        designs.append(self._format_design(X[idx_cheap], F[idx_cheap], problem, "Cost Optimized"))

        # 2. ✅ FIXED V3: Strength Optimized - Target 125-130% fc_target
        # Điều chỉnh linh hoạt dựa trên dữ liệu
        target_min = fc_target * 1.25  # 125% target
        target_max = fc_target * 1.30  # 130% target
        target_strength = (target_min + target_max) / 2  # Sweet spot
        
        # Chỉ xét designs >= fc_target
        valid_mask = actual_strengths >= fc_target
        
        if np.any(valid_mask):
            # Tìm design gần sweet spot nhất (125-130%)
            distances = np.abs(actual_strengths - target_strength)
            filtered_distances = np.where(valid_mask, distances, np.inf)
            idx_strong = np.argmin(filtered_distances)
            
            # Nếu design được chọn < 120% → lấy mạnh nhất thay thế
            if actual_strengths[idx_strong] < fc_target * 1.20:
                idx_strong = np.argmax(actual_strengths)
        else:
            # Fallback: lấy mạnh nhất
            idx_strong = np.argmax(actual_strengths)
        
        designs.append(self._format_design(X[idx_strong], F[idx_strong], problem, "Strength Optimized"))

        # 3. Eco-friendly - Lowest CO2
        idx_eco = np.argmin(F[:, 3])
        designs.append(self._format_design(X[idx_eco], F[idx_eco], problem, "Eco-friendly"))

        # 4. Balanced - Knee point
        idx_knee = self._find_knee_point(F)
        designs.append(self._format_design(X[idx_knee], F[idx_knee], problem, "Balanced"))

        # 5. Slump Optimized - Best slump accuracy
        idx_slump = np.argmin(F[:, 2])
        designs.append(self._format_design(X[idx_slump], F[idx_slump], problem, "Slump Optimized"))

        return designs

    def _actual_strengths_loop(self, X, problem, user_input) -> np.ndarray:
        """Bản gốc (từng dòng) - dùng làm fallback"""
        actual_strengths = []
        for i in range(len(X)):
            mix = problem._x_to_mix_dict(X[i])
            try:
                predictions = problem.predictor.predict_all(mix)
                predictions = problem.adjuster.adjust_predictions(
                    mix, predictions, problem.cement_type
                )
                age_target = user_input['age_target']
                if age_target == 28:
                    fc_age = predictions['f28']
                else:
                    fc_age = problem.predictor.predict_strength_at_age(mix, age_target)
                    fc_age = problem.adjuster.adjust_predictions(
                        mix, {'f28': fc_age}, problem.cement_type
                    )['f28']
                actual_strengths.append(fc_age)
            except Exception:
                actual_strengths.append(0)
        return np.array(actual_strengths)

    def _format_design(
        self,
        x: np.ndarray,
        f: np.ndarray,
        problem: ConcreteMixOptimizationProblem,
        profile: str
    ) -> Dict:
        """Format design thành Dict đầy đủ thông tin"""
        mix = problem._x_to_mix_dict(x)

        # Predictions
        predictions = problem.predictor.predict_all(mix)
        predictions = problem.adjuster.adjust_predictions(
            mix, predictions, problem.cement_type
        )

        # Cost & CO2
        cost_data = problem.cost_calc.calculate_total_cost(mix, problem.cement_type)
        co2_data = problem.co2_calc.calculate_total_emission(mix, problem.cement_type)

        # Validation
        is_valid, violations = problem.validate_mix(mix)

        # Score calculation
        f_norm = f.copy()
        f_norm[1] = -f_norm[1]
        
        f_scaled = np.array([
            f_norm[0] / 1000000,
            f_norm[1] / 50,
            f_norm[2] / 50,
            f_norm[3] / 500
        ])
        score = 1.0 / (1.0 + np.linalg.norm(f_scaled))

        return {
            'profile': profile,
            'cement_type': problem.cement_type,
            'mix_design': mix,
            'predictions': {
                'f28': predictions['f28'],
                's': predictions['s'],
                'slump': predictions['slump']
            },
            'objectives': {
                'cost': f[0],
                'strength': predictions['f28'],  # ✅ Dùng actual strength
                'slump_deviation': f[2],
                'co2': f[3]
            },
            'cost_breakdown': cost_data['breakdown'],
            'co2_breakdown': co2_data['breakdown'],
            'validation': {
                'is_valid': is_valid,
                'violations': violations
            },
            'score': score
        }

    def _find_knee_point(self, F: np.ndarray) -> int:
        """Tìm knee point trên Pareto front"""
        F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-10)
        distances = np.sqrt(np.sum(F_norm**2, axis=1))
        return np.argmin(distances)

    def _calculate_metrics(self, F: np.ndarray) -> Dict:
        """Tính metrics cho Pareto front"""
        return {
            'n_solutions': len(F),
            'cost_range': (float(F[:, 0].min()), float(F[:, 0].max())),
            'strength_range': (float(-F[:, 1].max()), float(-F[:, 1].min())),
            'co2_range': (float(F[:, 3].min()), float(F[:, 3].max())),
            'avg_slump_deviation': float(F[:, 2].mean())
        }


# ===== TEST =====
if __name__ == "__main__":
    print("✅ nsga2_optimizer.py V3 - Strength Optimized target 125-130%")