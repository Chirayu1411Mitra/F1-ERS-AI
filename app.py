from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # headless backend for server-side image generation
import matplotlib.pyplot as plt
import os
import sys
import subprocess

# Ensure FastF1 (auto-install if missing)
try:
    import fastf1
except Exception:  # pragma: no cover
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastf1", "--quiet"])  # non-interactive
        import fastf1  # type: ignore
    except Exception:
        fastf1 = None

app = Flask(__name__)

# Prepare FastF1 cache if available
if fastf1 is not None:
    os.makedirs(os.path.join(os.path.dirname(__file__), "fastf1_cache"), exist_ok=True)
    try:
        fastf1.Cache.enable_cache(os.path.join(os.path.dirname(__file__), "fastf1_cache"))
    except Exception:
        pass

def build_track_df(year:int, event:str, session_code:str="R", n_segments:int=10) -> pd.DataFrame:
    """Fetch telemetry for a session and bin into segments matching the notebook schema."""
    if fastf1 is None:
        raise RuntimeError("fastf1 is not installed in this environment.")
    # Normalize common short names to FastF1 event names
    aliases = {
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
    key = str(event).strip().upper()
    event_norm = aliases.get(key, event)
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
            rows.append({"length_m": float(r-l), "baseline_speed_kph": float(max(1.0, speed[0])),
                         "is_braking_zone": 0, "is_drs_zone": 0})
            continue
        seg_len = float(r - l)
        seg_speed = float(max(1.0, np.nanmean(speed[mask])))
        seg_brake = int(np.nanmean(brake[mask]) > 0.2)
        seg_drs = int(np.nanmean(drs_active[mask]) > 0.3)
        rows.append({"length_m": seg_len, "baseline_speed_kph": seg_speed,
                     "is_braking_zone": seg_brake, "is_drs_zone": seg_drs})
    df = pd.DataFrame(rows)
    df["length_m"] = df["length_m"].clip(lower=0.1)
    df["baseline_speed_kph"] = df["baseline_speed_kph"].clip(lower=1.0)
    return df

def generate_strategy(track_condition, drs_enabled):
    # Placeholder for AI optimization logic
    # In a real implementation, this would use the notebook's AI model
    # For now, we'll return dummy data
    lap_time = 80.5 + np.random.normal(0, 0.5)
    strategy = [
        "High ERS deployment in Sector 1",
        "Medium deployment through corners",
        "Full deployment on main straight"
    ]
    
    return lap_time, strategy

def create_visualization(x_mid: np.ndarray, speed_kph: np.ndarray, battery_mj: np.ndarray,
                         title_suffix: str) -> str:
    """Dual-axis plot of speed and battery over segment midpoints; return static path."""
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax1.plot(x_mid, speed_kph, 'b-o', label='Speed (KPH)')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Speed (KPH)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(x_mid, battery_mj, 'g--x', label='Battery (MJ)')
    ax2.set_ylabel('Battery (MJ)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    plt.title(f'Optimal ERS Strategy {title_suffix}')
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)
    filename = f"lap_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(os.path.join(static_dir, filename), bbox_inches='tight')
    plt.close()
    return f"/static/{filename}"

# --- GA/Simulation logic (mirrors notebook) ---
ACTION_COAST, ACTION_DEPLOY, ACTION_HARVEST = 0, 1, 2
MAX_BATTERY_MJ = 4.0
MAX_DEPLOY_PER_LAP_MJ = 4.0
MAX_HARVEST_PER_LAP_MJ = 2.0

def simulate_and_time(strategy, df, track_condition: str, drs_enabled: bool):
    grip_mod = 0.85 if track_condition == "WET" else 1.0
    harvest_eff = 0.7 if track_condition == "WET" else 1.0
    drs_mod = 1.15 if drs_enabled else 1.0
    total_time = 0.0
    current_batt = MAX_BATTERY_MJ
    total_deploy = 0.0
    total_harvest = 0.0
    n = len(df)
    DEPLOY_BOOST = 1.05
    HARVEST_BRAKE = 0.85
    cost_per_deploy = MAX_DEPLOY_PER_LAP_MJ / (n / 2)
    gain_per_harvest = MAX_HARVEST_PER_LAP_MJ / (n / 4)
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
        if speed_kph <= 0:
            return 1e9
        total_time += float(seg["length_m"]) / (speed_kph / 3.6)
    if total_deploy > MAX_DEPLOY_PER_LAP_MJ or total_harvest > MAX_HARVEST_PER_LAP_MJ:
        return 1e9
    return total_time

def simulate_full(strategy, df, track_condition: str, drs_enabled: bool):
    grip_mod = 0.85 if track_condition == "WET" else 1.0
    harvest_eff = 0.7 if track_condition == "WET" else 1.0
    drs_mod = 1.15 if drs_enabled else 1.0
    n = len(df)
    current_batt = MAX_BATTERY_MJ
    total_deploy = 0.0
    total_harvest = 0.0
    DEPLOY_BOOST = 1.05
    HARVEST_BRAKE = 0.85
    cost_per_deploy = MAX_DEPLOY_PER_LAP_MJ / (n / 2)
    gain_per_harvest = MAX_HARVEST_PER_LAP_MJ / (n / 4)
    speeds = []
    batteries = []
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
        speeds.append(speed_kph)
        batteries.append(current_batt)
    return np.array(speeds), np.array(batteries)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run-ai', methods=['POST'])
def run_ai():
    try:
        data = request.json or {}
        track_condition = str(data.get('track_condition', 'DRY')).upper()
        if track_condition not in {"DRY", "WET"}:
            track_condition = "DRY"
        drs_enabled = bool(data.get('drs_enabled', True))
        event = str(data.get('event', 'Monza')) or 'Monza'

        # Build track features from FastF1 for selected event (2024 Race)
        df = build_track_df(2024, event, 'R', n_segments=10)

        # Run GA optimizer (auto-install if missing)
        try:
            from geneticalgorithm import geneticalgorithm as ga
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "geneticalgorithm", "--quiet"])  # non-interactive
            from geneticalgorithm import geneticalgorithm as ga
        var_boundaries = np.array([[0, 2]] * len(df))
        def fitness(x):
            strat = [int(round(v)) for v in x]
            return simulate_and_time(strat, df, track_condition, drs_enabled)
        model = ga(function=fitness, dimension=len(df), variable_type='int',
                   variable_boundaries=var_boundaries,
                   algorithm_parameters={
                       'max_num_iteration': 250,
                       'population_size': 60,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.01,
                       'parents_portion': 0.8,
                       'crossover_probability': 0.5,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': 60
                   })
        model.run()
        best_strategy = [int(round(v)) for v in model.best_variable]
        best_time = float(model.best_function)

        # Re-simulate to get series
        speeds, batteries = simulate_full(best_strategy, df, track_condition, drs_enabled)
        lengths = df["length_m"].astype(float).to_numpy()
        cum = np.concatenate(([0.0], np.cumsum(lengths)))
        x_mid = (cum[1:] + cum[:-1]) / 2.0
        image_path = create_visualization(x_mid, speeds, batteries,
            title_suffix=f"({event} 2024, {track_condition}, DRS {'On' if drs_enabled else 'Off'})")

        label_map = {0: 'COAST', 1: 'DEPLOY', 2: 'HARVEST'}
        return jsonify({
            'lap_time': f'{best_time:.3f}',
            'strategy': [label_map[v] for v in best_strategy],
            'image_path': image_path
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)