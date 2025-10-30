from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load track data
track_data = pd.read_csv('track_data.csv')

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

def create_visualization(track_condition, drs_enabled):
    # Create a visualization of the strategy
    plt.figure(figsize=(10, 6))
    plt.plot(track_data['Distance'], track_data['Speed'], 'b-', label='Speed Profile')
    plt.title(f'Lap Visualization ({track_condition}, DRS {"Enabled" if drs_enabled else "Disabled"})')
    plt.xlabel('Distance (m)')
    plt.ylabel('Speed (km/h)')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_path = f'static/lap_viz_{timestamp}.png'
    plt.savefig(os.path.join(os.path.dirname(__file__), image_path))
    plt.close()
    
    return f'/{image_path}'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run-ai', methods=['POST'])
def run_ai():
    data = request.json
    track_condition = data.get('track_condition', 'DRY')
    drs_enabled = data.get('drs_enabled', True)
    
    # Generate optimization results
    lap_time, strategy = generate_strategy(track_condition, drs_enabled)
    
    # Create and save visualization
    image_path = create_visualization(track_condition, drs_enabled)
    
    return jsonify({
        'lap_time': f'{lap_time:.3f}',
        'strategy': strategy,
        'image_path': image_path
    })

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)