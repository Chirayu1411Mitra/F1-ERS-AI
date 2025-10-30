import os
import sys
import subprocess
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd


def ensure_fastf1() -> Any:  # returns module
    try:
        import fastf1  # type: ignore
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastf1", "--quiet"])  # non-interactive
        import fastf1  # type: ignore
    return fastf1


def ensure_geneticalgorithm():
    try:
        from geneticalgorithm import geneticalgorithm as ga  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "geneticalgorithm", "--quiet"])  # non-interactive


def normalize_event_name(event: str) -> str:
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
    return aliases.get(key, event)


# --- Simulation parameters (must match your app/notebook) ---
ACTION_COAST, ACTION_DEPLOY, ACTION_HARVEST = 0, 1, 2
MAX_BATTERY_MJ = 4.0
MAX_DEPLOY_PER_LAP_MJ = 4.0
MAX_HARVEST_PER_LAP_MJ = 2.0


def build_track_segments(fastf1_mod, year: int, event: str, session_code: str, n_segments: int) -> pd.DataFrame:
    session = fastf1_mod.get_session(year, event, session_code)
    session.load()
    lap = session.laps.pick_fastest()
    tel = lap.get_telemetry().dropna(subset=["Distance", "Speed"]).copy()

    distance = tel["Distance"].to_numpy()
    speed = tel["Speed"].to_numpy()
    # Brake flag
    brake = tel.get("Brake")
    if brake is None:
        t_s = tel["Time"].dt.total_seconds().to_numpy()
        dv = np.diff(speed, prepend=speed[0])
        dt = np.diff(t_s, prepend=t_s[0]) + 1e-6
        brake = (dv / dt < -2.0).astype(int)
    else:
        brake = brake.fillna(0).astype(int).to_numpy()
    # DRS active
    drs_raw = tel.get("DRS")
    drs_raw = (drs_raw.fillna(0).astype(int) if drs_raw is not None else pd.Series(0, index=tel.index))
    drs_active = (drs_raw >= 10).astype(int).to_numpy()

    edges = np.linspace(distance.min(), distance.max(), n_segments + 1)
    rows = []
    for i in range(n_segments):
        l, r = edges[i], edges[i + 1]
        mask = (distance >= l) & (distance < r if i < n_segments - 1 else distance <= r)
        if not np.any(mask):
            rows.append({
                "length_m": float(r - l),
                "baseline_speed_kph": float(max(1.0, speed[0])),
                "is_braking_zone": 0,
                "is_drs_zone": 0,
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
            "is_drs_zone": seg_drs,
        })
    df = pd.DataFrame(rows)
    df["length_m"] = df["length_m"].clip(lower=0.1)
    df["baseline_speed_kph"] = df["baseline_speed_kph"].clip(lower=1.0)
    return df


def simulate_and_time(strategy: List[int], df: pd.DataFrame, condition: str, drs_enabled: bool) -> float:
    grip_mod = 0.85 if condition == "WET" else 1.0
    harvest_eff = 0.7 if condition == "WET" else 1.0
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


def run_ga_optimal_strategy(df: pd.DataFrame, condition: str, drs_enabled: bool,
                            iterations: int, population: int) -> Dict[str, Any]:
    ensure_geneticalgorithm()
    from geneticalgorithm import geneticalgorithm as ga

    var_boundaries = np.array([[0, 2]] * len(df))

    def fitness(x):
        strat = [int(round(v)) for v in x]
        return simulate_and_time(strat, df, condition, drs_enabled)

    model = ga(
        function=fitness,
        dimension=len(df),
        variable_type='int',
        variable_boundaries=var_boundaries,
        algorithm_parameters={
            'max_num_iteration': iterations,
            'population_size': population,
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'parents_portion': 0.8,
            'crossover_probability': 0.5,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': max(25, iterations // 4)
        }
    )
    model.run()
    best_strategy = [int(round(v)) for v in model.best_variable]
    best_time = float(model.best_function)
    return {"strategy": best_strategy, "lap_time": best_time}


def write_row_wide(out_rows: List[Dict[str, Any]], meta: Dict[str, Any], df: pd.DataFrame,
                   best: Dict[str, Any]) -> None:
    row: Dict[str, Any] = dict(meta)
    n = len(df)
    lengths = df["length_m"].astype(float).to_numpy()
    speeds = df["baseline_speed_kph"].astype(float).to_numpy()
    brakes = df["is_braking_zone"].astype(int).to_numpy()
    drs = df["is_drs_zone"].astype(int).to_numpy()
    total_len = float(lengths.sum()) if n > 0 else 1.0
    cum_end = np.cumsum(lengths)  # end of segment distance

    # Global aggregates
    row["avg_speed_kph"] = float(np.mean(speeds)) if n else 0.0
    row["brake_ratio"] = float(np.mean(brakes)) if n else 0.0
    row["drs_ratio"] = float(np.mean(drs)) if n else 0.0

    for i in range(n):
        # Base features
        row[f"length_{i}"] = float(lengths[i])
        row[f"speed_{i}"] = float(speeds[i])
        row[f"brake_{i}"] = int(brakes[i])
        row[f"drs_{i}"] = int(drs[i])

        # Positional / geometry
        row[f"pos_{i}"] = float(i / (n - 1)) if n > 1 else 0.0
        row[f"cumdist_{i}"] = float(cum_end[i] / total_len)

        # Lookahead flags (next 1-2 segments)
        row[f"next1_brake_{i}"] = int(brakes[i + 1]) if i + 1 < n else 0
        row[f"next2_brake_{i}"] = int(brakes[i + 2]) if i + 2 < n else 0
        row[f"next1_drs_{i}"] = int(drs[i + 1]) if i + 1 < n else 0
        row[f"next2_drs_{i}"] = int(drs[i + 2]) if i + 2 < n else 0

        # Local speed dynamics
        prev_speed = speeds[i - 1] if i - 1 >= 0 else speeds[i]
        nxt_speed = speeds[i + 1] if i + 1 < n else speeds[i]
        row[f"speed_delta_{i}"] = float(speeds[i] - prev_speed)
        row[f"speed_ma3_{i}"] = float(np.mean([prev_speed, speeds[i], nxt_speed]))

        # Label
        row[f"action_{i}"] = int(best["strategy"][i])
    row["lap_time_best"] = float(best["lap_time"])
    out_rows.append(row)


def main():
    parser = argparse.ArgumentParser(description="Build ERS training dataset from FastF1 telemetry")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--events", nargs="*", default=[
        "Bahrain", "Jeddah", "Melbourne", "Imola", "Monaco",
        "Barcelona", "Silverstone", "Spa", "Monza", "Abu Dhabi"
    ])
    parser.add_argument("--session", type=str, default="R", help="FP1/FP2/FP3/Q/SQ/R")
    parser.add_argument("--n-segments", type=int, default=10)
    parser.add_argument("--condition", type=str, default="DRY", choices=["DRY", "WET"])
    parser.add_argument("--drs", type=int, default=1, help="1 enables DRS, 0 disables")
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--out", type=str, default=os.path.join("datasets", "ers_training.csv"))

    args = parser.parse_args()

    fastf1 = ensure_fastf1()
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fastf1_cache")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        fastf1.Cache.enable_cache(cache_dir)
    except Exception:
        pass

    out_rows: List[Dict[str, Any]] = []
    for ev in args.events:
        ev_name = normalize_event_name(ev)
        try:
            df = build_track_segments(fastf1, args.year, ev_name, args.session, args.n_segments)
            best = run_ga_optimal_strategy(df, args.condition, bool(args.drs), args.iterations, args.population)
            meta = {
                "year": int(args.year),
                "event": ev_name,
                "session": args.session,
                "condition": args.condition,
                "drs_enabled": int(bool(args.drs)),
                "n_segments": int(args.n_segments),
                "track_length_m": float(df["length_m"].sum()),
            }
            write_row_wide(out_rows, meta, df, best)
            print(f"Built row for {args.year} {ev_name}: lap_time={best['lap_time']:.3f}s")
        except Exception as e:
            print(f"Warning: skipping {args.year} {ev_name}: {e}")

    if not out_rows:
        print("No rows generated; nothing to write.")
        sys.exit(1)

    out_df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved {len(out_df)} rows to {args.out}")


if __name__ == "__main__":
    main()


