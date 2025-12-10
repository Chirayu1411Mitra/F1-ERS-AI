import fastf1
import json
import os
import numpy as np

# Tracks to pre-calculate
TRACKS = {
    "BAHRAIN": ("Bahrain", 2024),
    "JEDDAH": ("Saudi Arabian Grand Prix", 2024),
    "MELBOURNE": ("Australian Grand Prix", 2024),
    "IMOLA": ("Emilia Romagna Grand Prix", 2024),
    "MONACO": ("Monaco", 2024),
    "BARCELONA": ("Spanish Grand Prix", 2024),
    "SILVERSTONE": ("British Grand Prix", 2024),
    "SPA": ("Belgian Grand Prix", 2024),
    "MONZA": ("Italian Grand Prix", 2024),
    "ZANDVOORT": ("Dutch Grand Prix", 2024),
    "BAKU": ("Azerbaijan Grand Prix", 2024),
    "SINGAPORE": ("Singapore Grand Prix", 2024),
    "AUSTIN": ("United States Grand Prix", 2024),
    "MEXICO": ("Mexico City Grand Prix", 2024),
    "BRAZIL": ("São Paulo Grand Prix", 2024),
    "VEGAS": ("Las Vegas Grand Prix", 2023), # 2024 might not be available yet or same layout
    "QATAR": ("Qatar Grand Prix", 2023),
    "ABU DHABI": ("Abu Dhabi Grand Prix", 2023)
}

OUTPUT_FILE = os.path.join("datasets", "track_geometries.json")

def generate():
    print(f"Generating track geometries to {OUTPUT_FILE}...")
    os.makedirs("datasets", exist_ok=True)
    
    geometries = {}
    
    for key, (event_name, year) in TRACKS.items():
        print(f"Processing {key} ({event_name} {year})...")
        try:
            session = fastf1.get_session(year, event_name, 'R')
            session.load()
            lap = session.laps.pick_fastest()
            tel = lap.get_telemetry().dropna(subset=['X', 'Y'])
            
            x = tel['X'].to_numpy()
            y = tel['Y'].to_numpy()
            
            # Downsample to save space (every ~10th point is usually enough for a map)
            # We want roughly 500-1000 points per track
            step = max(1, len(x) // 800)
            x_small = x[::step]
            y_small = y[::step]
            
            # Normalize to 0-100 range immediately to save compute on frontend/backend later
            min_x, max_x = x_small.min(), x_small.max()
            min_y, max_y = y_small.min(), y_small.max()
            
            # Store simply as lists
            geometries[event_name] = {
                "x": [round(float(v), 1) for v in x_small],
                "y": [round(float(v), 1) for v in y_small],
                "dist": [round(float(v), 1) for v in tel['Distance'].to_numpy()[::step]]
            }
            print(f"  -> Success ({len(x_small)} points)")
            
        except Exception as e:
            print(f"  -> Failed: {e}")
            
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(geometries, f)
    
    print("Done! File saved.")

if __name__ == "__main__":
    generate()
