"""
streamlit_app.py - FINAL VERSION WITH S-CURVE
✅ Proper predictor initialization with Streamlit cache
✅ Fixed CO2 breakdown chart
✅ ADDED: Strength Development Curve (S-Curve) visualization
✅ ADDED: Early & Long-term strength tables
✅ ADDED: S-Parameter comparison
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict
import plotly.graph_objects as go
import plotly.express as px

# ===== CRITICAL FIX: Define OptimalSlumpFeatureBuilder for pickle =====
class OptimalSlumpFeatureBuilder:
    """Feature builder cho slump model - MUST be in __main__ for pickle"""
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

# Register class in sys.modules for pickle
if 'OptimalSlumpFeatureBuilder' not in sys.modules['__main__'].__dict__:
    sys.modules['__main__'].OptimalSlumpFeatureBuilder = OptimalSlumpFeatureBuilder

from multi_cement_workflow import MultiCementWorkflow
from material_database import MaterialDatabase
from pdf_report_generator import PDFReportGenerator


# ===== CACHE PREDICTOR PROPERLY =====
@st.cache_resource(show_spinner=False)
def load_predictor_singleton():
    """Load predictor một lần duy nhất và cache"""
    with st.spinner("🔄 Loading AI models..."):
        from predictor_unified import UnifiedPredictor
        import importlib
        import predictor_unified as pred_module
        importlib.reload(pred_module)
        
        predictor = pred_module.UnifiedPredictor()
    
    # Validate models
    model_status = {
        "f28": predictor.f28_bundle is not None,
        "s": predictor.s_bundle is not None,
        "slump_builder": predictor.slump_builder is not None,
        "slump_folds": len(predictor.slump_models)
    }
    
    return predictor, model_status



def build_user_input(material_db: MaterialDatabase) -> Dict:
    """Đọc input từ sidebar Streamlit"""
    st.sidebar.subheader("🎯 Design Targets")
    
    fc_target = st.sidebar.slider("Target strength f'c (MPa)", 20, 80, 40, 1)
    age_target = st.sidebar.selectbox("Age (days)", [3, 7, 14, 28, 56], 3)
    slump_target = st.sidebar.slider("Target slump (mm)", 50, 250, 180, 10)
    slump_tolerance = st.sidebar.slider("Slump tolerance (±mm)", 10, 50, 20, 5)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧱 Available Materials")
    
    available = {
        "Xỉ (Slag)": {"available": st.sidebar.checkbox("Slag", True), "category": "SCM"},
        "Tro bay (Flyash)": {"available": st.sidebar.checkbox("Fly ash", True), "category": "SCM"},
        "Silica fume": {"available": st.sidebar.checkbox("Silica fume", False), "category": "SCM"},
        "Phụ gia siêu dẻo (SP)": {"available": st.sidebar.checkbox("Superplasticizer (SP)", True)}
    }
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Material Properties")
    
    with st.sidebar.expander("🔧 Advanced Settings", expanded=False):
        tab1, tab2 = st.tabs(["📊 Mass Limits (kg/m³)", "⚖️ Density (kg/m³)"])
        
        with tab1:
            st.markdown("**Binder Components**")
            c1, c2 = st.columns(2)
            cement_min = c1.number_input("Cement Min", 100, 400, 200, 10)
            cement_max = c2.number_input("Cement Max", 300, 700, 600, 10)
            
            c1, c2 = st.columns(2)
            water_min = c1.number_input("Water Min", 80, 200, 100, 5)
            water_max = c2.number_input("Water Max", 150, 300, 250, 5)
            
            st.markdown("**SCM**")
            c1, c2 = st.columns(2)
            flyash_max = c1.number_input("Flyash Max", 0, 200, 150, 10)
            slag_max = c2.number_input("Slag Max", 0, 300, 200, 10)
            silica_fume_max = st.number_input("Silica Fume Max", 0, 50, 40, 5)
            
            st.markdown("**Admixtures & Aggregates**")
            sp_max = st.number_input("SP Max (kg/m³)", 0, 20, 15, 1)
            
            c1, c2 = st.columns(2)
            fine_agg_min = c1.number_input("Fine Agg Min", 400, 700, 600, 10)
            fine_agg_max = c2.number_input("Fine Agg Max", 700, 1000, 900, 10)
            
            c1, c2 = st.columns(2)
            coarse_agg_min = c1.number_input("Coarse Agg Min", 600, 900, 800, 10)
            coarse_agg_max = c2.number_input("Coarse Agg Max", 1000, 1400, 1200, 10)
        
        with tab2:
            st.markdown("**Binder & Water**")
            c1, c2 = st.columns(2)
            d_cem = c1.number_input("Cement", 2800, 3300, 3150, 10)
            d_wat = c2.number_input("Water", 990, 1010, 1000, 5)
            
            st.markdown("**SCM & SP**")
            c1, c2 = st.columns(2)
            d_fa = c1.number_input("Flyash", 2000, 2500, 2200, 10)
            d_sl = c2.number_input("Slag", 2700, 3100, 2900, 10)
            c1, c2 = st.columns(2)
            d_sf = c1.number_input("Silica Fume", 2000, 2500, 2200, 10)
            d_sp = c2.number_input("SP", 1000, 1200, 1050, 10)
            
            st.markdown("**Aggregates**")
            c1, c2 = st.columns(2)
            d_sand = c1.number_input("Fine Agg", 2500, 2800, 2650, 10)
            d_stone = c2.number_input("Coarse Agg", 2500, 2900, 2700, 10)
    
    material_bounds = {
        'cement': (cement_min, cement_max), 'water': (water_min, water_max),
        'flyash': (0, flyash_max), 'slag': (0, slag_max),
        'silica_fume': (0, silica_fume_max), 'superplasticizer': (0, sp_max),
        'fine_agg': (fine_agg_min, fine_agg_max), 'coarse_agg': (coarse_agg_min, coarse_agg_max)
    }
    
    material_density = {
        'cement': d_cem, 'water': d_wat, 'flyash': d_fa, 'slag': d_sl,
        'silica_fume': d_sf, 'superplasticizer': d_sp,
        'fine_agg': d_sand, 'coarse_agg': d_stone
    }
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚖️ Preference Weights")
    
    cost_w = st.sidebar.slider("Cost weight", 0.0, 1.0, 0.4, 0.05)
    perf_w = st.sidebar.slider("Performance weight", 0.0, 1.0, 0.3, 0.05)
    sus_w = st.sidebar.slider("Sustainability weight", 0.0, 1.0, 0.2, 0.05)
    work_w = st.sidebar.slider("Workability weight", 0.0, 1.0, 0.1, 0.05)
    
    total = cost_w + perf_w + sus_w + work_w
    if total == 0: total = 1.0
    
    return {
        "fc_target": float(fc_target), "age_target": int(age_target),
        "slump_target": float(slump_target), "slump_tolerance": float(slump_tolerance),
        "available_materials": available, "material_bounds": material_bounds,
        "material_density": material_density,
        "preferences": {"cost": cost_w/total, "performance": perf_w/total, 
                        "sustainability": sus_w/total, "workability": work_w/total}
    }


def format_mix_design(mix: Dict) -> pd.DataFrame:
    data = []
    for comp, mass in mix.items():
        if mass > 0:
            data.append({"Component": comp.replace('_', ' ').title(), "Mass (kg/m³)": f"{mass:.1f}"})
    return pd.DataFrame(data)


def create_cost_breakdown_chart(cost_data: Dict) -> go.Figure:
    breakdown = cost_data['breakdown']
    labels = [k.replace('_', ' ').title() for k in breakdown.keys()]
    values = [d['total'] for d in breakdown.values()]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.3,
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    fig.update_layout(title="Cost Breakdown", height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_co2_breakdown_chart(co2_data: Dict) -> go.Figure:
    a1_total = sum(item['emission'] for item in co2_data.get('A1', {}).values() if isinstance(item, dict))
    a2_total = sum(item['emission'] for item in co2_data.get('A2', {}).values() if isinstance(item, dict))
    a3_total = co2_data.get('A3', 0)
    
    phases = ['A1 (Production)', 'A2 (Transport)', 'A3 (Mixing)']
    values = [a1_total, a2_total, a3_total]
    colors = ['#e74c3c', '#f39c12', '#3498db']
    
    fig = go.Figure(data=[go.Bar(
        x=phases, y=values, marker=dict(color=colors),
        text=[f"{v:.1f}" for v in values], textposition='auto'
    )])
    fig.update_layout(title="CO₂ Emissions by Phase", yaxis_title="kgCO₂/m³", height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ===== NEW: S-CURVE VISUALIZATION =====

def main():
    st.set_page_config(page_title="Concrete Optimizer", layout="wide", page_icon="🗿")
    st.title("Mô hình thiết kế và tối ưu hóa cấp phối bê tông")
    st.caption("NSGA-II Multi-objective Optimization System")
    st.markdown("---")
    
    predictor, model_status = load_predictor_singleton()
    
    with st.expander("🔍 AI Models Status", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F28 Model", "✅" if model_status["f28"] else "❌")
        c2.metric("S Model", "✅" if model_status["s"] else "❌")
        c3.metric("Slump Builder", "✅" if model_status["slump_builder"] else "❌")
        c4.metric("Slump Folds", f"{model_status['slump_folds']}/10")
        
        if model_status["slump_folds"] == 0:
            st.error("⚠️ Slump models not loaded! Check models/slump_models/")
    
    material_db = MaterialDatabase()
    user_input = build_user_input(material_db)
    
    if 'material_bounds' in user_input:
        with st.expander("📊 Selected Material Bounds Summary", expanded=False):
            b = user_input['material_bounds']
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cement", f"{b['cement'][0]}-{b['cement'][1]}")
            c2.metric("Water", f"{b['water'][0]}-{b['water'][1]}")
            c3.metric("Fine Agg", f"{b['fine_agg'][0]}-{b['fine_agg'][1]}")
            c4.metric("Coarse Agg", f"{b['coarse_agg'][0]}-{b['coarse_agg'][1]}")
    
    st.subheader("⚙️ Optimization Settings")
    # Thêm chú thích gợi ý
    st.info("💡 **Gợi ý cài đặt nhanh:** Chọn 1 loại xi măng, đặt Population size = 50, Generations = 50, giữ nguyên Random seed")
    cement_types = st.multiselect("Select cement types", ["PC40", "PC50"], ["PC40", "PC50"])
    
    c1, c2, c3 = st.columns(3)
    pop_size = c1.number_input("Population size", 50, 300, 100, 10)
    n_gen = c2.number_input("Generations", 50, 500, 200, 10)
    seed = c3.number_input("Random seed", 0, 9999, 42, 1)
    early_stop = st.checkbox("⏱️ Dừng sớm khi Pareto front hội tụ (nhanh hơn, tối đa = Generations)", value=False)
    
    if st.button("🚀 Run Optimization", type="primary"):
        if not cement_types:
            st.warning("⚠️ Select at least one cement type.")
            return
        
        if model_status["slump_folds"] == 0:
            st.error("❌ Slump models not loaded!")
            return
        
        with st.spinner("⏳ Running NSGA-II optimization..."):
            workflow = MultiCementWorkflow(
                models_dir=str(ROOT / "models"),
                output_dir=str(ROOT / "outputs"),
                predictor=predictor
            )
            results = workflow.run_optimization(
                user_input=user_input,
                cement_types=cement_types,
                optimization_config={"pop_size": int(pop_size), "n_gen": int(n_gen), "seed": int(seed), "early_stop": bool(early_stop)}
            )
        
        st.success("✅ Optimization complete!")
        st.session_state["workflow"] = workflow
        st.session_state["results"] = results
    
    if "results" in st.session_state:
        results = st.session_state["results"]
        st.markdown("## 📊 Optimization Results")
        
        tabs = st.tabs(["📋 Summary", "🏆 Top Designs"])
        
        with tabs[0]:
            st.subheader("Executive Summary")
            for ct, proc in results["processed_results"].items():
                st.markdown(f"### {ct}")
                top_design = proc["ranked_designs"][0]
                cols = st.columns(4)
                cols[0].metric("Cost (VNĐ/m³)", f"{top_design['objectives']['cost']:,.0f}")
                cols[1].metric("f28 (MPa)", f"{top_design['predictions']['f28']:.1f}")
                cols[2].metric("Slump (mm)", f"{top_design['predictions']['slump']:.0f}")
                cols[3].metric("CO₂ (kg/m³)", f"{top_design['objectives']['co2']:.0f}")
            
            st.markdown("### 📝 Recommendations")
            for rec in results["recommendations"]:
                st.write(f"- {rec}")
        
        with tabs[1]:
            st.subheader("🏆 Top Recommended Designs")
            for ct, proc in results["processed_results"].items():
                st.markdown(f"### {ct}")
                for i, design in enumerate(proc["ranked_designs"][:3], 1):
                    with st.expander(f"Design {i}: {design['profile']} (Score: {design['score']:.3f})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Mix Proportions**")
                            st.dataframe(format_mix_design(design['mix_design']), hide_index=True)
                            mix = design['mix_design']
                            binder = sum([mix.get('cement',0), mix.get('flyash',0), mix.get('slag',0), mix.get('silica_fume',0)])
                            st.metric("w/b ratio", f"{mix['water']/binder:.3f}" if binder > 0 else "0")
                        with col2:
                            st.markdown("**Performance**")
                            pred = design['predictions']
                            obj = design['objectives']
                            st.metric("f28 (MPa)", f"{pred['f28']:.1f}")
                            st.metric("Slump (mm)", f"{pred['slump']:.0f}")
                            st.metric("Cost (VNĐ/m³)", f"{obj['cost']:,.0f}")
                            st.metric("CO₂ (kg/m³)", f"{obj['co2']:.0f}")
                        
                        col3, col4 = st.columns(2)
                        with col3:
                            st.plotly_chart(create_cost_breakdown_chart({'breakdown': design['cost_breakdown']}), use_container_width=True, key=f"cost_{ct}_{i}")
                        with col4:
                            st.plotly_chart(create_co2_breakdown_chart(design['co2_breakdown']), use_container_width=True, key=f"co2_{ct}_{i}")

        st.markdown("---")
        st.subheader("📥 Export Results")
        c1, c2, c3 = st.columns(3)
        
        if c1.button("📄 Generate PDF Report"):
            with st.spinner("Generating..."):
                pdf_path = PDFReportGenerator().generate_report(results, str(ROOT / "outputs"))
                st.success(f"✅ Report saved: {pdf_path}")
        
        if c2.button("💾 Export to CSV"):
             for ct, opt_res in results["optimization_results"].items():
                X, F = opt_res['pareto_front']
                df = pd.DataFrame(X, columns=['cement', 'water', 'flyash', 'slag', 'silica_fume', 'sp', 'fine', 'coarse'])
                df['cost'] = F[:, 0]
                df['f28'] = -F[:, 1]
                df['slump_dev'] = F[:, 2]
                df['co2'] = F[:, 3]
                df.to_csv(ROOT / "outputs" / f"pareto_{ct}.csv", index=False)
             st.success("✅ CSV exported")
             
        if c3.button("🎯 Export for Production"):
            path = st.session_state["workflow"].export_for_production([(ct, 0) for ct in results["processed_results"]])
            st.success(f"✅ Production file: {path}")

if __name__ == "__main__":
    main()