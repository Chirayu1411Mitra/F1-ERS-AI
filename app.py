from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # headless backend for server-side image generation
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib import colors as mcolors
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

def _build_track_segments_xy(event: str, year: int, session_code: str, n_segments: int, strategy: list[int]):
    """Return normalized SVG-like segments for interactive 2D track: list of {points:[(x,y)], action:int}.
    Coordinates are normalized to [0,100] in both axes maintaining aspect ratio when rendered with viewBox 0 0 100 100.
    """
    if fastf1 is None:
        return []
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
    try:
        key = str(event).strip().upper()
        event_norm = aliases.get(key, event)
        session = fastf1.get_session(year, event_norm, session_code)
        session.load()
        lap = session.laps.pick_fastest()
        tel = lap.get_telemetry().dropna(subset=["Distance"]).copy()
        if ("X" not in tel.columns) or ("Y" not in tel.columns):
            return []
        dist = tel["Distance"].to_numpy()
        x = tel["X"].to_numpy()
        y = tel["Y"].to_numpy()
        # Normalize to 0..100 box
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        xr = max(1e-6, xmax - xmin)
        yr = max(1e-6, ymax - ymin)
        # Build edges consistent with binning
        edges = np.linspace(dist.min(), dist.max(), n_segments + 1)
        segments = []
        for i in range(n_segments):
            l, r = edges[i], edges[i + 1]
            mask = (dist >= l) & (dist < r if i < n_segments - 1 else dist <= r)
            if not np.any(mask):
                segments.append({"points": [], "action": int(strategy[i]) if i < len(strategy) else 0})
                continue
            # Downsample points to keep payload reasonable
            idx = np.where(mask)[0]
            step = max(1, len(idx) // 50)
            idx = idx[::step]
            pts = []
            for j in idx:
                px = (x[j] - xmin) / xr * 100.0
                py = (y[j] - ymin) / yr * 100.0
                pts.append([round(px, 2), round(py, 2)])
            segments.append({"points": pts, "action": int(strategy[i]) if i < len(strategy) else 0})
        return segments
    except Exception:
        return []

def create_track_map_image(event: str, year: int, session_code: str, n_segments: int, strategy: list[int]) -> str:
    """Render a 2D track map using FastF1 XY telemetry and color segments by strategy.
    Returns static path to the saved PNG.
    """
    if fastf1 is None:
        raise RuntimeError("fastf1 is not installed in this environment.")
    # Normalize event as in build_track_df
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
    tel = lap.get_telemetry().dropna(subset=["Distance"]).copy()
    # Some datasets may not include X/Y; guard gracefully
    if ("X" not in tel.columns) or ("Y" not in tel.columns):
        raise RuntimeError("Telemetry does not include XY coordinates for track map.")

    dist = tel["Distance"].to_numpy()
    x = tel["X"].to_numpy()
    y = tel["Y"].to_numpy()

    # Build edges consistent with build_track_df binning
    edges = np.linspace(dist.min(), dist.max(), n_segments + 1)

    # Colors for actions
    color_map = {
        0: "#5b6b7c",  # COAST
        1: "#e74c3c",  # DEPLOY
        2: "#27ae60",  # HARVEST
    }

    fig = plt.figure(figsize=(8, 8), facecolor="#0f1318")
    ax = plt.gca()
    ax.set_facecolor("#0f1318")
    # Plot base track lightly (background line)
    ax.plot(x, y, color="#0b0e13", linewidth=8, alpha=0.9, zorder=1, solid_capstyle='round')
    # Overlay colored segments with thin black outline for separation
    for i in range(n_segments):
        l, r = edges[i], edges[i + 1]
        mask = (dist >= l) & (dist < r if i < n_segments - 1 else dist <= r)
        if not np.any(mask):
            continue
        base_color = color_map.get(int(strategy[i]), "#888888")
        glow_rgba = mcolors.to_rgba(base_color, alpha=0.35)
        # Outline path
        ax.plot(
            x[mask], y[mask], color="#0b0b0b", linewidth=6, zorder=2, solid_capstyle='round'
        )
        # Colored path on top
        line, = ax.plot(
            x[mask], y[mask], color=base_color, linewidth=4, zorder=3, solid_capstyle='round'
        )
        line.set_path_effects([
            patheffects.Stroke(linewidth=10, foreground=glow_rgba),
            patheffects.Normal(),
        ])

    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')
    plt.title(f"{event_norm}", color="#cfd8e3")

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)
    filename = f"track_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(os.path.join(static_dir, filename), bbox_inches='tight', pad_inches=0.1, facecolor=fig.get_facecolor())
    plt.close()
    return f"/static/{filename}"

@app.route('/track-xy', methods=['POST'])
def track_xy():
    try:
        if fastf1 is None:
            return jsonify({"error": "fastf1 not available"}), 500
        data = request.json or {}
        event = str(data.get('event', 'Monza')) or 'Monza'
        # Normalize as used elsewhere
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
        session = fastf1.get_session(2024, event_norm, 'R')
        session.load()
        lap = session.laps.pick_fastest()
        tel = lap.get_telemetry()
        if 'X' not in tel.columns or 'Y' not in tel.columns:
            return jsonify({"error": "XY not available"}), 400
        x = tel['X'].to_numpy()
        y = tel['Y'].to_numpy()
        # Normalize to [0,1] with padding while preserving aspect
        min_x, max_x = float(np.nanmin(x)), float(np.nanmax(x))
        min_y, max_y = float(np.nanmin(y)), float(np.nanmax(y))
        width = max(1e-6, max_x - min_x)
        height = max(1e-6, max_y - min_y)
        pad = 0.06
        if width >= height:
            scale = 1.0 - 2 * pad
            norm_x = (x - min_x) / width * scale + pad
            norm_y = (y - min_y) / width * scale + (1 - (height / width) * scale) / 2
        else:
            scale = 1.0 - 2 * pad
            norm_y = (y - min_y) / height * scale + pad
            norm_x = (x - min_x) / height * scale + (1 - (width / height) * scale) / 2
        # Return arrays (may be large; client will sample for animation)
        return jsonify({
            'x': norm_x.tolist(),
            'y': norm_y.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- GA/Simulation logic (mirrors notebook) ---
ACTION_COAST, ACTION_DEPLOY, ACTION_HARVEST = 0, 1, 2
MAX_BATTERY_MJ = 4.0
MAX_DEPLOY_PER_LAP_MJ = 4.0
MAX_HARVEST_PER_LAP_MJ = 2.0

def simulate_and_time(strategy, df, track_condition: str, drs_enabled: bool,
                      aero_downforce: float = 0.5, drag_coeff: float = 1.0,
                      wind_speed_mps: float = 0.0, wind_is_headwind: bool = True):
    # Condition modifiers
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
    cost_per_deploy = MAX_DEPLOY_PER_LAP_MJ / (n / 2)
    gain_per_harvest = MAX_HARVEST_PER_LAP_MJ / (n / 4)
    # Simplified aerodynamics/wind effects
    downforce_bonus = 1.0 + 0.06 * max(0.0, min(1.0, aero_downforce))  # helps in corners
    straight_drag_penalty = 1.0 / max(0.6, min(1.4, drag_coeff))        # hurts on straights
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
        # Apply aero/wind factors: assume corners align with braking zones
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
    cost_per_deploy = MAX_DEPLOY_PER_LAP_MJ / (n / 2)
    gain_per_harvest = MAX_HARVEST_PER_LAP_MJ / (n / 4)
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


# -------- Policy model loading/inference --------
_POLICY_CACHE = None

def _ensure_joblib():
    try:
        import joblib  # type: ignore
        return joblib
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib", "scikit-learn", "--quiet"])  # noqa: E501
        import joblib  # type: ignore
        return joblib


def load_policy_model(model_path: str = os.path.join(os.path.dirname(__file__), "models", "ers_policy.pkl")):
    global _POLICY_CACHE
    if _POLICY_CACHE is not None:
        return _POLICY_CACHE
    joblib = _ensure_joblib()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Policy model not found at {model_path}. Train it first with scripts/train_policy.py")
    _POLICY_CACHE = joblib.load(model_path)
    return _POLICY_CACHE


def _build_policy_features(df: pd.DataFrame, track_condition: str, drs_enabled: bool, feature_names: list) -> np.ndarray:
    """Create a feature vector aligned with saved feature_names from the trained model."""
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
        if prefix == "pos_":
            return float(idx / (n - 1)) if n > 1 and idx < n else 0.0
        if prefix == "cumdist_":
            return float(cum_end[idx] / total_len) if idx < n and total_len > 0 else 0.0
        if prefix == "next1_brake_":
            j = idx + 1
            return float(int(brakes[j])) if j < n else 0.0
        if prefix == "next2_brake_":
            j = idx + 2
            return float(int(brakes[j])) if j < n else 0.0
        if prefix == "next1_drs_":
            j = idx + 1
            return float(int(drs[j])) if j < n else 0.0
        if prefix == "next2_drs_":
            j = idx + 2
            return float(int(drs[j])) if j < n else 0.0
        if prefix == "speed_delta_":
            prev = speeds[idx - 1] if idx - 1 >= 0 else (speeds[idx] if idx < n else 0.0)
            cur = speeds[idx] if idx < n else 0.0
            return float(cur - prev)
        if prefix == "speed_ma3_":
            prev = speeds[idx - 1] if idx - 1 >= 0 else (speeds[idx] if idx < n else 0.0)
            cur = speeds[idx] if idx < n else 0.0
            nxt = speeds[idx + 1] if idx + 1 < n else (speeds[idx] if idx < n else 0.0)
            return float(np.mean([prev, cur, nxt]))
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
            # Segment features with pattern prefix + index
            try:
                prefix, idx_str = name.rsplit('_', 1)
                idx = int(idx_str)
                prefix = prefix + '_'  # restore trailing underscore
            except Exception:
                values.append(0.0)
                continue
            values.append(seg_val(prefix, idx))

    return np.array(values, dtype=float).reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run-ai', methods=['POST'])
def run_ai():
    try:
        data = request.json or {}
        track_condition = str(data.get('track_condition', 'DRY')).upper()
        if track_condition not in {"DRY", "INTERMEDIATE", "WET"}:
            track_condition = "DRY"
        drs_enabled = bool(data.get('drs_enabled', True))
        event = str(data.get('event', 'Monza')) or 'Monza'
        use_policy = bool(data.get('use_policy', False))
        aero_downforce = float(data.get('aero_downforce', 0.5))
        drag_coeff = float(data.get('drag_coeff', 1.0))
        wind_speed_mps = float(data.get('wind_speed_mps', 0.0))
        wind_is_headwind = bool(data.get('wind_is_headwind', True))

        # Build track features from FastF1 for selected event (2024 Race)
        df = build_track_df(2024, event, 'R', n_segments=10)

        policy_error = None
        if use_policy:
            # Use trained policy
            try:
                model_bundle = load_policy_model()
                clf = model_bundle["model"]
                feat_names = model_bundle.get("feature_names", [])
                X = _build_policy_features(df, track_condition, drs_enabled, feat_names)
                y_pred = clf.predict(X)[0]
                best_strategy = [int(v) for v in y_pred]
                best_time = float(simulate_and_time(best_strategy, df, track_condition, drs_enabled,
                                                   aero_downforce, drag_coeff, wind_speed_mps, wind_is_headwind))
                mode_tag = 'Policy'
            except FileNotFoundError as e:
                # Fallback to GA seamlessly
                policy_error = str(e)
                use_policy = False
        else:
            # Run GA optimizer (auto-install if missing)
            try:
                from geneticalgorithm import geneticalgorithm as ga
            except Exception:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "geneticalgorithm", "--quiet"])  # non-interactive
                from geneticalgorithm import geneticalgorithm as ga
            var_boundaries = np.array([[0, 2]] * len(df))
            def fitness(x):
                strat = [int(round(v)) for v in x]
                return simulate_and_time(strat, df, track_condition, drs_enabled,
                                         aero_downforce, drag_coeff, wind_speed_mps, wind_is_headwind)
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
            mode_tag = 'GA'

        # Re-simulate to get series
        speeds, batteries = simulate_full(best_strategy, df, track_condition, drs_enabled,
                                          aero_downforce, drag_coeff, wind_speed_mps, wind_is_headwind)
        lengths = df["length_m"].astype(float).to_numpy()
        cum = np.concatenate(([0.0], np.cumsum(lengths)))
        x_mid = (cum[1:] + cum[:-1]) / 2.0
        title_suffix = f"({mode_tag} • {event} 2024, {track_condition}, DRS {'On' if drs_enabled else 'Off'})"
        image_path = create_visualization(x_mid, speeds, batteries,
            title_suffix=title_suffix)

        label_map = {0: 'COAST', 1: 'DEPLOY', 2: 'HARVEST'}
        track_segments = _build_track_segments_xy(event, 2024, 'R', len(df), best_strategy)
        # Robustness: evaluate same strategy across conditions
        robustness = {}
        for cond in ["DRY", "INTERMEDIATE", "WET"]:
            robustness[cond] = float(simulate_and_time(best_strategy, df, cond, drs_enabled,
                                                      aero_downforce, drag_coeff, wind_speed_mps, wind_is_headwind))
        # Per-segment details for UI
        details = []
        battery_before = MAX_BATTERY_MJ
        for i in range(len(df)):
            seg = df.iloc[i]
            length_m = float(seg["length_m"])
            baseline_kph = float(seg["baseline_speed_kph"])
            final_kph = float(speeds[i])
            time_s = length_m / (final_kph / 3.6) if final_kph > 0 else float('inf')
            details.append({
                "index": i + 1,
                "action": label_map.get(best_strategy[i], 'COAST'),
                "length_m": round(length_m, 2),
                "baseline_speed_kph": round(baseline_kph, 1),
                "final_speed_kph": round(final_kph, 1),
                "time_s": round(time_s, 3),
                "battery_before_mj": round(battery_before, 3),
                "battery_after_mj": round(float(batteries[i]), 3),
                "is_drs_zone": int(seg["is_drs_zone"]) == 1,
                "is_braking_zone": int(seg["is_braking_zone"]) == 1,
            })
            battery_before = float(batteries[i])
        resp = {
            'lap_time': f'{best_time:.3f}',
            'strategy': [label_map[v] for v in best_strategy],
            'image_path': image_path,
            'source': mode_tag,
            'track_segments': track_segments,
            'segment_details': details,
            'robustness': robustness,
            'series': {
                'distance_m': [float(v) for v in x_mid.tolist()],
                'speed_kph': [float(v) for v in speeds.tolist()],
                'battery_mj': [float(v) for v in batteries.tolist()],
            },
            'chart_title': f'Optimal ERS Strategy {title_suffix}'
        }
        if policy_error:
            resp['note'] = f'Policy unavailable: {policy_error}. Used GA fallback.'
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict-policy', methods=['POST'])
def predict_policy():
    try:
        data = request.json or {}
        track_condition = str(data.get('track_condition', 'DRY')).upper()
        if track_condition not in {"DRY", "INTERMEDIATE", "WET"}:
            track_condition = "DRY"
        drs_enabled = bool(data.get('drs_enabled', True))
        event = str(data.get('event', 'Monza')) or 'Monza'
        aero_downforce = float(data.get('aero_downforce', 0.5))
        drag_coeff = float(data.get('drag_coeff', 1.0))
        wind_speed_mps = float(data.get('wind_speed_mps', 0.0))
        wind_is_headwind = bool(data.get('wind_is_headwind', True))

        # Build features from telemetry
        df = build_track_df(2024, event, 'R', n_segments=10)
        model_bundle = load_policy_model()
        clf = model_bundle["model"]

        feat_names = model_bundle.get("feature_names", [])
        X = _build_policy_features(df, track_condition, drs_enabled, feat_names)
        y_pred = clf.predict(X)[0]
        best_strategy = [int(v) for v in y_pred]
        best_time = float(simulate_and_time(best_strategy, df, track_condition, drs_enabled))

        # Series for plotting
        speeds, batteries = simulate_full(best_strategy, df, track_condition, drs_enabled,
                                          aero_downforce, drag_coeff, wind_speed_mps, wind_is_headwind)
        lengths = df["length_m"].astype(float).to_numpy()
        cum = np.concatenate(([0.0], np.cumsum(lengths)))
        x_mid = (cum[1:] + cum[:-1]) / 2.0
        title_suffix = f"(Policy • {event} 2024, {track_condition}, DRS {'On' if drs_enabled else 'Off'})"
        image_path = create_visualization(x_mid, speeds, batteries,
            title_suffix=title_suffix)

        # Create 2D interactive data and image
        track_map_path = create_track_map_image(event=event, year=2024, session_code='R',
                                               n_segments=len(df), strategy=best_strategy)
        track_segments = _build_track_segments_xy(event, 2024, 'R', len(df), best_strategy)
        robustness = {}
        for cond in ["DRY", "INTERMEDIATE", "WET"]:
            robustness[cond] = float(simulate_and_time(best_strategy, df, cond, drs_enabled,
                                                      aero_downforce, drag_coeff, wind_speed_mps, wind_is_headwind))
        # Per-segment details
        label_map = {0: 'COAST', 1: 'DEPLOY', 2: 'HARVEST'}
        details = []
        battery_before = MAX_BATTERY_MJ
        for i in range(len(df)):
            seg = df.iloc[i]
            length_m = float(seg["length_m"])
            baseline_kph = float(seg["baseline_speed_kph"])
            final_kph = float(speeds[i])
            time_s = length_m / (final_kph / 3.6) if final_kph > 0 else float('inf')
            details.append({
                "index": i + 1,
                "action": label_map.get(best_strategy[i], 'COAST'),
                "length_m": round(length_m, 2),
                "baseline_speed_kph": round(baseline_kph, 1),
                "final_speed_kph": round(final_kph, 1),
                "time_s": round(time_s, 3),
                "battery_before_mj": round(battery_before, 3),
                "battery_after_mj": round(float(batteries[i]), 3),
                "is_drs_zone": int(seg["is_drs_zone"]) == 1,
                "is_braking_zone": int(seg["is_braking_zone"]) == 1,
            })
            battery_before = float(batteries[i])

        return jsonify({
            'source': 'policy',
            'lap_time': f'{best_time:.3f}',
            'strategy': [label_map[v] for v in best_strategy],
            'image_path': image_path,
            'track_image_path': track_map_path,
            'track_segments': track_segments,
            'segment_details': details,
            'robustness': robustness,
            'series': {
                'distance_m': [float(v) for v in x_mid.tolist()],
                'speed_kph': [float(v) for v in speeds.tolist()],
                'battery_mj': [float(v) for v in batteries.tolist()],
            },
            'chart_title': f'Optimal ERS Strategy {title_suffix}'
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)