import os
import sys
import argparse
import subprocess
from typing import List, Tuple

import numpy as np
import pandas as pd


def ensure_sklearn():
    try:
        import sklearn  # type: ignore
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "joblib", "--quiet"])  # non-interactive
    finally:
        from sklearn.multioutput import MultiOutputClassifier  # noqa: F401
        from sklearn.ensemble import RandomForestClassifier  # noqa: F401
        import joblib  # noqa: F401


def build_feature_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # Infer segments from columns like length_0...
    seg_cols = sorted([c for c in df.columns if c.startswith("length_")], key=lambda x: int(x.split("_")[1]))
    n_segments = len(seg_cols)

    feat_names: List[str] = []
    X_parts: List[np.ndarray] = []
    # Core bases always expected
    core_bases = ("length_", "speed_", "brake_", "drs_")
    # Optional engineered bases (include only if present)
    opt_bases = ("pos_", "cumdist_", "next1_brake_", "next2_brake_", "next1_drs_", "next2_drs_", "speed_delta_", "speed_ma3_")
    bases = list(core_bases) + [b for b in opt_bases]
    for i in range(n_segments):
        for base in bases:
            col = f"{base}{i}"
            if col not in df.columns:
                if base in core_bases:
                    raise ValueError(f"Missing column '{col}' in dataset")
                else:
                    continue  # skip optional feature if not found
            feat_names.append(col)
            X_parts.append(df[col].to_numpy().reshape(-1, 1))

    # Globals
    globals_list = ["track_length_m", "drs_enabled", "avg_speed_kph", "brake_ratio", "drs_ratio"]
    for g in globals_list:
        if g in df.columns:
            feat_names.append(g)
            X_parts.append(df[g].to_numpy().reshape(-1, 1))

    # Condition one-hot
    cond = df.get("condition")
    if cond is not None:
        cond_vals = cond.astype(str).str.upper()
        cond_dry = (cond_vals == "DRY").astype(int).to_numpy().reshape(-1, 1)
        cond_wet = (cond_vals == "WET").astype(int).to_numpy().reshape(-1, 1)
        feat_names += ["cond_DRY", "cond_WET"]
        X_parts += [cond_dry, cond_wet]

    X = np.hstack(X_parts)

    # Targets: action_0..N-1
    y_parts: List[np.ndarray] = []
    for i in range(n_segments):
        col = f"action_{i}"
        if col not in df.columns:
            raise ValueError(f"Missing label column '{col}' in dataset")
        y_parts.append(df[col].astype(int).to_numpy().reshape(-1, 1))
    y = np.hstack(y_parts)
    return X, y, feat_names


def split_by_event(df: pd.DataFrame, test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    events = sorted(df["event"].astype(str).unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(events)
    k = max(1, int(len(events) * test_ratio))
    test_events = set(events[:k])
    df_train = df[~df["event"].isin(test_events)].reset_index(drop=True)
    df_test = df[df["event"].isin(test_events)].reset_index(drop=True)
    return df_train, df_test


def main():
    ensure_sklearn()
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import joblib

    parser = argparse.ArgumentParser(description="Train a multi-output ERS policy model from ers_training.csv")
    parser.add_argument("--input", type=str, default=os.path.join("datasets", "ers_training.csv"))
    parser.add_argument("--out", type=str, default=os.path.join("models", "ers_policy.pkl"))
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=12)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Dataset not found: {args.input}")

    df = pd.read_csv(args.input)
    if "event" not in df.columns:
        raise ValueError("Dataset must include 'event' column for group-wise split")

    df_train, df_test = split_by_event(df, args.test_ratio, args.seed)
    X_train, y_train, feat_names = build_feature_targets(df_train)
    X_test, y_test, _ = build_feature_targets(df_test)

    clf = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            n_jobs=-1,
            random_state=args.seed,
        )
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    # Report accuracy per segment and mean
    seg_acc = []
    for i in range(y_test.shape[1]):
        acc = float(accuracy_score(y_test[:, i], y_pred[:, i]))
        seg_acc.append(acc)
    mean_acc = float(np.mean(seg_acc))
    print("Per-segment accuracy:", [round(a, 3) for a in seg_acc])
    print("Mean accuracy:", round(mean_acc, 3))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({
        "model": clf,
        "feature_names": feat_names,
        "mean_accuracy": mean_acc,
    }, args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()


