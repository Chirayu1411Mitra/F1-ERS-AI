"""
F1 ERS AI - Hugging Face Spaces Version
Optimized for deployment on Hugging Face Spaces with Gradio interface
"""

import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib import colors as mcolors
import os
import sys
import subprocess
import threading
import time
import io
from pathlib import Path

# ========== Auto-install dependencies ==========
def _ensure_package(package_name, import_name=None):
    """Ensure a package is installed"""
    import_name = import_name or package_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
            return True
        except:
            return False

# Install FastF1 if needed
fastf1 = None
try:
    import fastf1
except:
    if _ensure_package("fastf1"):
        try:
            import fastf1
        except:
            fastf1 = None

# ========== Configuration ==========
CACHE_DIR = Path("fastf1_cache")
STATIC_DIR = Path("static")
MODELS_DIR = Path("models")

for dir_path in [CACHE_DIR, STATIC_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Enable FastF1 cache
if fastf1 is not None:
    try:
        fastf1.Cache.enable_cache(str(CACHE_DIR))
    except:
        pass

# ========== Constants ==========
ACTION_COAST, ACTION_DEPLOY, ACTION_HARVEST = 0, 1, 2
MAX_BATTERY_MJ = 4.0
MAX_DEPLOY_PER_LAP_MJ = 4.0
MAX_HARVEST_PER_LAP_MJ = 2.0

# Track aliases for normalization
TRACK_ALIASES = {
    "BAHRAIN": "Bahrain",
    "JEDDAH": "Saudi Arabian Grand Prix",
    "MELBOURNE": "Australian Grand Prix",
    "IMOLA": "Emilia Romagna Grand Prix",
    "MONACO": "Monaco",
    "BARCELONA": "Spanish Grand Prix",
    "SILVERSTONE": "British Grand Prix",
    "SPA": "Belgian Grand Prix",
    "MONZA": "Italian Grand Prix",
    "ABU DHABI": "Abu Dhabi Grand Prix",
}

# ========== Track data builder ==========
def build_track_df(year: int, event: str, session_code: str = "R", n_segments: int = 10) -> pd.DataFrame:
    """Fetch telemetry for a session and bin into segments"""
    if fastf1 is None:
        raise RuntimeError("fastf1 is not available. Cannot fetch track data.")
    
    key = str(event).strip().upper()
    event_norm = TRACK_ALIASES.get(key, event)
    
    try:
        session = fastf1.get_session(year, event_norm, session_code)
        session.load()
        lap = session.laps.pick_fastest()
        tel = lap.get_telemetry().dropna(subset=["Distance", "Speed"]).copy()

        distance = tel["Distance"].to_numpy()
        speed = tel["Speed"].to_numpy()
        brake = tel.get("Brake")
        
        if brake is None:
            t_s = tel["Time"].dt.total_seconds().to_numpy()
            dv = np.diff(speed, prepend=speed[0])
            dt = np.diff(t_s, prepend=t_s[0]) + 1e-6
            brake = (dv/dt < -2.0).astype(int)
        else:
            brake = brake.fillna(0).astype(int).to_numpy()
        
        drs_raw = tel.get("DRS")
        drs_raw = (drs_raw.fillna(0).astype(int) if drs_raw is not None else pd.Series(0, index=tel.index))
        drs_active = (drs_raw >= 10).astype(int).to_numpy()

        edges = np.linspace(distance.min(), distance.max(), n_segments + 1)
        rows = []
        
        for i in range(n_segments):
            l, r = edges[i], edges[i+1]
            mask = (distance >= l) & (distance < r if i < n_segments-1 else distance <= r)
            
            if not np.any(mask):
                rows.append({
                    "length_m": float(r-l),
                    "baseline_speed_kph": float(max(1.0, speed[0])),
                    "is_braking_zone": 0,
                    "is_drs_zone": 0
                })
                continue
            
            seg_len = float(r - l)
            seg_speed = float(max(1.0, np.nanmean(speed[mask])))
            seg_brake = int(np.nanmean(brake[mask]) > 0.2)
            seg_drs = int(np.nanmean(drs_active[mask]) > 0.3)
            
            rows.append({
                "length_m": seg_len,
                "baseline_speed_kph": seg_speed,
                "is_braking_zone": seg_brake,
                "is_drs_zone": seg_drs
            })
        
        df = pd.DataFrame(rows)
        df["length_m"] = df["length_m"].clip(lower=0.1)
        df["baseline_speed_kph"] = df["baseline_speed_kph"].clip(lower=1.0)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch track data: {str(e)}")

# ========== Simulation functions ==========
def simulate_and_time(strategy, df, track_condition: str, drs_enabled: bool,
                      aero_downforce: float = 0.5, drag_coeff: float = 1.0,
                      wind_speed_mps: float = 0.0, wind_is_headwind: bool = True):
    """Simulate and return lap time"""
    
    if track_condition == "WET":
        grip_mod, harvest_eff = 0.85, 0.70
    elif track_condition == "INTERMEDIATE":
        grip_mod, harvest_eff = 0.93, 0.85
    else:
        grip_mod, harvest_eff = 1.00, 1.00
    
    drs_mod = 1.15 if drs_enabled else 1.0
    total_time = 0.0
    current_batt = MAX_BATTERY_MJ
    total_deploy = 0.0
    total_harvest = 0.0
    n = len(df)
    
    DEPLOY_BOOST = 1.05
    HARVEST_BRAKE = 0.85
    cost_per_deploy = MAX_DEPLOY_PER_LAP_MJ / (n / 2) if n > 0 else 1.0
    gain_per_harvest = MAX_HARVEST_PER_LAP_MJ / (n / 4) if n > 0 else 1.0
    
    downforce_bonus = 1.0 + 0.06 * max(0.0, min(1.0, aero_downforce))
    straight_drag_penalty = 1.0 / max(0.6, min(1.4, drag_coeff))
    wind_k = 0.003 * max(0.0, wind_speed_mps)
    wind_factor_straight = (1.0 - wind_k) if wind_is_headwind else (1.0 + wind_k)

    for i in range(n):
        seg = df.iloc[i]
        speed_kph = float(seg["baseline_speed_kph"]) * grip_mod
        
        if seg["is_drs_zone"] == 1 and drs_mod > 1.0:
            speed_kph *= drs_mod
        
        act = strategy[i]
        
        if act == ACTION_DEPLOY:
            if current_batt > cost_per_deploy and total_deploy < MAX_DEPLOY_PER_LAP_MJ:
                speed_kph *= DEPLOY_BOOST
                current_batt -= cost_per_deploy
                total_deploy += cost_per_deploy
        
        elif act == ACTION_HARVEST:
            if seg["is_braking_zone"] == 1 and current_batt < MAX_BATTERY_MJ and total_harvest < MAX_HARVEST_PER_LAP_MJ:
                speed_kph *= HARVEST_BRAKE
                delta = gain_per_harvest * harvest_eff
                current_batt = min(MAX_BATTERY_MJ, current_batt + delta)
                total_harvest += delta
        
        if int(seg["is_braking_zone"]) == 1:
            speed_kph *= downforce_bonus
        else:
            speed_kph *= straight_drag_penalty * wind_factor_straight

        if speed_kph <= 0:
            return 1e9
        
        total_time += float(seg["length_m"]) / (speed_kph / 3.6)
    
    if total_deploy > MAX_DEPLOY_PER_LAP_MJ or total_harvest > MAX_HARVEST_PER_LAP_MJ:
        return 1e9
    
    return total_time

def simulate_full(strategy, df, track_condition: str, drs_enabled: bool,
                  aero_downforce: float = 0.5, drag_coeff: float = 1.0,
                  wind_speed_mps: float = 0.0, wind_is_headwind: bool = True):
    """Simulate and return detailed data"""
    
    if track_condition == "WET":
        grip_mod, harvest_eff = 0.85, 0.70
    elif track_condition == "INTERMEDIATE":
        grip_mod, harvest_eff = 0.93, 0.85
    else:
        grip_mod, harvest_eff = 1.00, 1.00
    
    drs_mod = 1.15 if drs_enabled else 1.0
    n = len(df)
    current_batt = MAX_BATTERY_MJ
    total_deploy = 0.0
    total_harvest = 0.0
    
    DEPLOY_BOOST = 1.05
    HARVEST_BRAKE = 0.85
    cost_per_deploy = MAX_DEPLOY_PER_LAP_MJ / (n / 2) if n > 0 else 1.0
    gain_per_harvest = MAX_HARVEST_PER_LAP_MJ / (n / 4) if n > 0 else 1.0
    
    speeds = []
    batteries = []
    downforce_bonus = 1.0 + 0.06 * max(0.0, min(1.0, aero_downforce))
    straight_drag_penalty = 1.0 / max(0.6, min(1.4, drag_coeff))
    wind_k = 0.003 * max(0.0, wind_speed_mps)
    wind_factor_straight = (1.0 - wind_k) if wind_is_headwind else (1.0 + wind_k)

    for i in range(n):
        seg = df.iloc[i]
        speed_kph = float(seg["baseline_speed_kph"]) * grip_mod
        
        if seg["is_drs_zone"] == 1 and drs_mod > 1.0:
            speed_kph *= drs_mod
        
        act = strategy[i]
        
        if act == ACTION_DEPLOY and current_batt > cost_per_deploy and total_deploy < MAX_DEPLOY_PER_LAP_MJ:
            speed_kph *= DEPLOY_BOOST
            current_batt -= cost_per_deploy
            total_deploy += cost_per_deploy
        
        elif act == ACTION_HARVEST and seg["is_braking_zone"] == 1 and current_batt < MAX_BATTERY_MJ and total_harvest < MAX_HARVEST_PER_LAP_MJ:
            speed_kph *= HARVEST_BRAKE
            delta = gain_per_harvest * harvest_eff
            current_batt = min(MAX_BATTERY_MJ, current_batt + delta)
            total_harvest += delta
        
        if int(seg["is_braking_zone"]) == 1:
            speed_kph *= downforce_bonus
        else:
            speed_kph *= straight_drag_penalty * wind_factor_straight
        
        speeds.append(speed_kph)
        batteries.append(current_batt)
    
    return np.array(speeds), np.array(batteries)

# ========== Visualization ==========
def create_visualization(x_mid: np.ndarray, speed_kph: np.ndarray, battery_mj: np.ndarray,
                        title_suffix: str):
    """Create dual-axis plot and return as PIL Image"""
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(x_mid, speed_kph, 'b-o', label='Speed (KPH)', linewidth=2, markersize=4)
    ax1.set_xlabel('Distance (m)', fontsize=12)
    ax1.set_ylabel('Speed (KPH)', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(x_mid, battery_mj, 'g--x', label='Battery (MJ)', linewidth=2, markersize=6)
    ax2.set_ylabel('Battery (MJ)', color='g', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='g')
    
    plt.title(f'Optimal ERS Strategy {title_suffix}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    from PIL import Image
    return Image.open(buf)

# ========== Policy model loading ==========
_POLICY_CACHE = None

def _ensure_joblib():
    """Ensure joblib is installed"""
    try:
        import joblib
        return joblib
    except:
        _ensure_package("joblib")
        _ensure_package("scikit-learn")
        try:
            import joblib
            return joblib
        except:
            return None

def load_policy_model(model_path: str = "models/ers_policy.pkl"):
    """Load trained policy model"""
    global _POLICY_CACHE
    if _POLICY_CACHE is not None:
        return _POLICY_CACHE
    
    joblib = _ensure_joblib()
    if joblib is None:
        return None
    
    if not os.path.exists(model_path):
        return None
    
    try:
        _POLICY_CACHE = joblib.load(model_path)
        return _POLICY_CACHE
    except:
        return None

# ========== Gradio Interface ==========
def optimize_ers_strategy(
    event: str,
    track_condition: str,
    drs_enabled: bool,
    use_ai_model: bool,
    aero_downforce: float = 0.5,
    drag_coeff: float = 1.0,
    wind_speed_mps: float = 0.0,
    wind_is_headwind: bool = True
):
    """Main optimization function for Gradio"""
    
    try:
        # Fetch track data
        df = build_track_df(2024, event, 'R', n_segments=10)
        
        # Generate strategy
        if use_ai_model:
            model_bundle = load_policy_model()
            if model_bundle is not None:
                try:
                    clf = model_bundle["model"]
                    feat_names = model_bundle.get("feature_names", [])
                    X = _build_policy_features(df, track_condition, drs_enabled, feat_names)
                    y_pred = clf.predict(X)[0]
                    best_strategy = [int(v) for v in y_pred]
                    source = "🤖 AI Model"
                except:
                    best_strategy = _generate_ga_strategy(df, track_condition, drs_enabled)
                    source = "🧬 GA Optimizer"
            else:
                best_strategy = _generate_ga_strategy(df, track_condition, drs_enabled)
                source = "🧬 GA Optimizer"
        else:
            best_strategy = _generate_ga_strategy(df, track_condition, drs_enabled)
            source = "🧬 GA Optimizer"
        
        # Simulate
        best_time = float(simulate_and_time(best_strategy, df, track_condition, drs_enabled,
                                           aero_downforce, drag_coeff, wind_speed_mps, wind_is_headwind))
        
        speeds, batteries = simulate_full(best_strategy, df, track_condition, drs_enabled,
                                         aero_downforce, drag_coeff, wind_speed_mps, wind_is_headwind)
        
        # Visualization
        lengths = df["length_m"].astype(float).to_numpy()
        cum = np.concatenate(([0.0], np.cumsum(lengths)))
        x_mid = (cum[1:] + cum[:-1]) / 2.0
        
        title_suffix = f"({source} • {event} 2024, {track_condition}, DRS {'✓' if drs_enabled else '✗'})"
        image = create_visualization(x_mid, speeds, batteries, title_suffix=title_suffix)
        
        # Generate results table
        label_map = {0: '🔵 COAST', 1: '🔴 DEPLOY', 2: '🟢 HARVEST'}
        details = []
        battery_before = MAX_BATTERY_MJ
        
        for i in range(len(df)):
            seg = df.iloc[i]
            length_m = float(seg["length_m"])
            baseline_kph = float(seg["baseline_speed_kph"])
            final_kph = float(speeds[i])
            time_s = length_m / (final_kph / 3.6) if final_kph > 0 else float('inf')
            
            details.append({
                "Segment": i + 1,
                "Action": label_map.get(best_strategy[i], '🔵 COAST'),
                "Length (m)": f"{length_m:.2f}",
                "Base Speed (kph)": f"{baseline_kph:.1f}",
                "Final Speed (kph)": f"{final_kph:.1f}",
                "Time (s)": f"{time_s:.3f}",
                "Battery Before (MJ)": f"{battery_before:.3f}",
                "Battery After (MJ)": f"{batteries[i]:.3f}",
            })
            battery_before = float(batteries[i])
        
        # Create robustness analysis
        robustness_results = []
        for cond in ["DRY", "INTERMEDIATE", "WET"]:
            time_val = float(simulate_and_time(best_strategy, df, cond, drs_enabled,
                                             aero_downforce, drag_coeff, wind_speed_mps, wind_is_headwind))
            robustness_results.append({
                "Condition": cond,
                "Lap Time (s)": f"{time_val:.3f}"
            })
        
        results_text = f"""
        ## 🏁 Results
        
        **Lap Time:** {best_time:.3f} seconds  
        **Strategy Source:** {source}  
        **Track:** {event} 2024  
        **Weather:** {track_condition}  
        **DRS:** {'Enabled ✓' if drs_enabled else 'Disabled ✗'}
        """
        
        return image, pd.DataFrame(details), pd.DataFrame(robustness_results), results_text
    
    except Exception as e:
        error_image = create_error_image(str(e))
        return error_image, pd.DataFrame(), pd.DataFrame(), f"❌ Error: {str(e)}"

def _generate_ga_strategy(df, track_condition, drs_enabled):
    """Generate strategy using Genetic Algorithm"""
    try:
        from geneticalgorithm import geneticalgorithm as ga
    except:
        _ensure_package("geneticalgorithm")
        from geneticalgorithm import geneticalgorithm as ga
    
    var_boundaries = np.array([[0, 2]] * len(df))
    
    def fitness(x):
        strat = [int(round(v)) for v in x]
        return simulate_and_time(strat, df, track_condition, drs_enabled)
    
    model = ga(
        function=fitness,
        dimension=len(df),
        variable_type='int',
        variable_boundaries=var_boundaries,
        algorithm_parameters={
            'max_num_iteration': 200,
            'population_size': 50,
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'parents_portion': 0.8,
            'crossover_probability': 0.5,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': 50
        }
    )
    model.run()
    return [int(round(v)) for v in model.best_variable]

def _build_policy_features(df: pd.DataFrame, track_condition: str, drs_enabled: bool, feature_names: list) -> np.ndarray:
    """Build feature vector for policy model"""
    n = len(df)
    lengths = df["length_m"].astype(float).to_numpy()
    speeds = df["baseline_speed_kph"].astype(float).to_numpy()
    brakes = df["is_braking_zone"].astype(int).to_numpy()
    drs = df["is_drs_zone"].astype(int).to_numpy()
    
    total_len = float(lengths.sum()) if n > 0 else 1.0
    cum_end = np.cumsum(lengths) if n else np.array([])
    avg_speed_kph = float(np.mean(speeds)) if n else 0.0
    brake_ratio = float(np.mean(brakes)) if n else 0.0
    drs_ratio = float(np.mean(drs)) if n else 0.0

    def seg_val(prefix: str, idx: int) -> float:
        if prefix == "length_":
            return float(lengths[idx]) if idx < n else 0.0
        if prefix == "speed_":
            return float(speeds[idx]) if idx < n else 0.0
        if prefix == "brake_":
            return float(int(brakes[idx])) if idx < n else 0.0
        if prefix == "drs_":
            return float(int(drs[idx])) if idx < n else 0.0
        return 0.0

    values = []
    for name in feature_names:
        if name == "track_length_m":
            values.append(total_len)
        elif name == "drs_enabled":
            values.append(1.0 if drs_enabled else 0.0)
        elif name == "avg_speed_kph":
            values.append(avg_speed_kph)
        elif name == "brake_ratio":
            values.append(brake_ratio)
        elif name == "drs_ratio":
            values.append(drs_ratio)
        elif name == "cond_DRY":
            values.append(1.0 if track_condition == "DRY" else 0.0)
        elif name == "cond_WET":
            values.append(1.0 if track_condition == "WET" else 0.0)
        else:
            try:
                prefix, idx_str = name.rsplit('_', 1)
                idx = int(idx_str)
                prefix = prefix + '_'
                values.append(seg_val(prefix, idx))
            except:
                values.append(0.0)

    return np.array(values, dtype=float).reshape(1, -1)

def create_error_image(error_msg):
    """Create error visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, f"❌ Error\n\n{error_msg}", 
            ha='center', va='center', fontsize=14, wrap=True,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    from PIL import Image
    return Image.open(buf)

# ========== Gradio Interface Setup ==========
with gr.Blocks(title="F1 ERS AI - Hugging Face", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏎️ F1 ERS AI - Energy Recovery Strategy Optimizer")
    gr.Markdown("Optimize your Formula 1 car's ERS deployment strategy using AI and Genetic Algorithms")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuration")
            
            event = gr.Dropdown(
                choices=list(TRACK_ALIASES.keys()),
                value="MONZA",
                label="🏁 Track",
                info="Select F1 circuit"
            )
            
            track_condition = gr.Radio(
                choices=["DRY", "INTERMEDIATE", "WET"],
                value="DRY",
                label="🌦️ Track Condition",
                info="Current weather"
            )
            
            drs_enabled = gr.Checkbox(
                value=True,
                label="🚀 DRS Enabled",
                info="Enable Drag Reduction System"
            )
            
            use_ai_model = gr.Checkbox(
                value=False,
                label="🤖 Use AI Model",
                info="Use trained AI model (if available)"
            )
            
            with gr.Group(label="Advanced Parameters"):
                aero_downforce = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.1,
                    label="🔧 Aerodynamic Downforce",
                    info="Wing adjustment (0.0-1.0)"
                )
                drag_coeff = gr.Slider(
                    0.6, 1.4, value=1.0, step=0.05,
                    label="⚡ Drag Coefficient",
                    info="Aerodynamic drag (0.6-1.4)"
                )
                wind_speed_mps = gr.Slider(
                    0.0, 10.0, value=0.0, step=0.5,
                    label="💨 Wind Speed (m/s)",
                    info="Wind strength"
                )
                wind_is_headwind = gr.Checkbox(
                    value=True,
                    label="🌬️ Headwind",
                    info="Is wind a headwind?"
                )
            
            run_button = gr.Button(
                "🚀 Optimize Strategy",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            
            plot_output = gr.Image(label="📈 Strategy Visualization")
            
            results_text = gr.Markdown("Click 'Optimize Strategy' to begin")
            
            with gr.Tabs():
                with gr.TabItem("📋 Segment Details"):
                    details_table = gr.Dataframe(
                        label="Segment-by-segment breakdown",
                        interactive=False
                    )
                
                with gr.TabItem("🎯 Robustness"):
                    robustness_table = gr.Dataframe(
                        label="Performance across conditions",
                        interactive=False
                    )
    
    # Connect button to function
    run_button.click(
        fn=optimize_ers_strategy,
        inputs=[event, track_condition, drs_enabled, use_ai_model, aero_downforce, drag_coeff, wind_speed_mps, wind_is_headwind],
        outputs=[plot_output, details_table, robustness_table, results_text],
        api_name="optimize"
    )
    
    gr.Markdown("""
    ---
    ### 📚 About This Project
    
    This is an AI-powered ERS (Energy Recovery System) optimization tool for Formula 1.
    
    **Features:**
    - 🧬 Genetic Algorithm optimization
    - 🤖 Pre-trained AI model support
    - 📊 Real-time telemetry analysis
    - 🌍 Support for all F1 tracks
    - 🎛️ Advanced parameter tuning
    
    **Deployment:** [GitHub](https://github.com/Chirayu1411Mitra/F1-ERS-AI)
    """)

# ========== Launch Gradio App ==========
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        debug=False
    )
