# backend/app.py - Enhanced with trajectory tracking and SimPy time sync

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import threading
import time
import os
import sys
import math
import random

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_data import SIMULATION_CONFIGS
from integrated_simulation import SimulationManager
from entities.servers.offloading_algorithme import PPOModel
app = Flask(__name__)

def validate_socket_origin(origin, environ=None):
    """Validate socket.io origins by checking environ headers directly (no Flask request context needed)"""
    # Check if behind proxy by looking at environ headers directly
    forwarded_for = environ.get('HTTP_X_FORWARDED_FOR') if environ else None
    forwarded_proto = environ.get('HTTP_X_FORWARDED_PROTO') if environ else None
    
    # If behind proxy (nginx), accept any origin - no validation needed
    if forwarded_for or forwarded_proto:
        return True
    
    # Local dev only - restrict to localhost
    return origin in ['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost', 'http://127.0.0.1']

# CORS: Accept any origin (nginx will handle access control)
CORS(app, origins=['*'], supports_credentials=True)

# SocketIO with environ-based validation
socketio = SocketIO(
    app, 
    cors_allowed_origins=validate_socket_origin,
    async_mode='threading',
    ping_timeout=10,
    ping_interval=5
)

# Global simulation manager with real simulation integration
simulation_manager = SimulationManager(socketio)
PPO_MODEL = PPOModel()
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    print("[API] Health check requested")
    return jsonify({
        "status": "healthy",
        "message": "Edge Computing Simulation API - Real Simulation Integrated",
        "version": "2.1",
        "features": ["real_simulation", "trajectory_tracking", "simpy_time_sync", "15_cities", "configurable_edge_servers"]
    })

@app.route('/api/config/options', methods=['GET'])
def get_config_options():
    """Get available configuration options for dropdowns"""
    print("[API] Configuration options requested")
    try:
        # Load cities from Location.json
        cities_path = os.path.join(os.path.dirname(__file__), 'Data', 'locations.json')
        if os.path.exists(cities_path):
            with open(cities_path, 'r') as f:
                cities_data = json.load(f)

            cities_config = []
            for city_data in cities_data:
                cities_config.append({
                    "value": city_data["city"].lower().replace(" ", "_"),
                    "label": f"{city_data['city']}",
                    "center": [city_data["latitude"], city_data["longitude"]]
                })
            
            # Update SIMULATION_CONFIGS with loaded cities
            SIMULATION_CONFIGS["cities"] = cities_config
            
        # Add edge server count options for mobility scenario
        SIMULATION_CONFIGS["edge_servers"] = [
            {"value": 2, "label": "2 Edge Servers"},
            {"value": 3, "label": "3 Edge Servers"},
            {"value": 4, "label": "4 Edge Servers"}
        ]
        
        return jsonify(SIMULATION_CONFIGS)
    except Exception as e:
        print(f"[API] Error loading config options: {e}")
        return jsonify({"error": "Failed to load configuration options"}), 500

@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """Start real simulation with given configuration"""
    try:
        config = request.json
        print(f"[API] Start simulation requested with config: {config}")

        # Validate required fields
        required_fields = ['scenario', 'vehicles', 'duration', 'city']
        for field in required_fields:
            if field not in config:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Add optional fields with defaults
        config.setdefault('task_rate', 10)
        config.setdefault('fps', 1)
        config.setdefault('vehicle_speed', 50)
        config.setdefault('edge_coverage', 2.0)
        config.setdefault('edge_servers', 1)  # Default to 1 for general scenario

        # Start the real simulation
        result = simulation_manager.start_simulation(config)
        
        if result["success"]:
            print(f"[API] Real simulation started successfully: {result['simulation_id']}")
            
            # Emit layout change trigger
            socketio.emit('layout_change_trigger', {
                'show_simulation': True,
                'simulation_id': result["simulation_id"]
            })
            
            return jsonify({
                "message": "Real simulation started successfully",
                "simulation_id": result["simulation_id"],
                "config": config,
                "type": "real_simulation"
            })
        else:
            print(f"[API] Failed to start simulation: {result['error']}")
            return jsonify({"error": result["error"]}), 400

    except Exception as e:
        print(f"[API] Exception starting simulation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    """Stop current real simulation"""
    try:
        print("[API] Stop simulation requested")
        result = simulation_manager.stop_simulation()
        
        # Emit layout change trigger to hide simulation
        socketio.emit('layout_change_trigger', {
            'show_simulation': False
        })
        
        if result["success"]:
            print("[API] Simulation stopped successfully")
        else:
            print(f"[API] Failed to stop simulation: {result.get('error', 'Unknown error')}")
            
        return jsonify(result)
    except Exception as e:
        print(f"[API] Exception stopping simulation: {e}")
        return jsonify({"error": str(e)}), 500

# WebSocket event handlers remain the same...
@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print('[WebSocket] Client connected')
    emit('connected', {
        'message': 'Connected to Edge Computing Simulation - Enhanced Version 2.1',
        'version': '2.1',
        'features': ['trajectory_tracking', 'simpy_time_sync', '15_cities']
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print('[WebSocket] Client disconnected')

@socketio.on('subscribe_updates')
def handle_subscribe():
    """Handle client subscription to real-time updates"""
    print('[WebSocket] Client subscribed to real-time tracker updates')
    emit('subscribed', {
        'message': 'Subscribed to real-time simulation updates with trajectory tracking',
        'update_frequency': '10ms'
    })
    
def print_startup_banner():
    """Print startup banner with configuration info"""
    print("\n" + "="*70)
    print("EDGE COMPUTING SIMULATION - ENHANCED VERSION 2.1")
    print("="*70)
    print("✓ Real EdgeComputingSimulation integrated")
    print("✓ Vehicle trajectory tracking (planned + traveled)")
    print("✓ SimPy time synchronization")
    print("✓ 15 cities from Location.json")
    print("✓ Configurable edge servers (2-4 for mobility)")
    print("✓ Enhanced WebSocket real-time communication")
    print("="*70)
    
    # Load and display cities
    try:
        cities_path = os.path.join(os.path.dirname(__file__), 'Data', 'locations.json')
        if os.path.exists(cities_path):
            with open(cities_path, 'r') as f:
                cities_data = json.load(f)
            print("Available Cities:")
            for i, city in enumerate(cities_data):
                marker = "●" if i < 5 else "○"
                print(f"  {marker} {city['city']:15} [{city['latitude']:8.4f}, {city['longitude']:8.4f}]")
            print(f"  Total: {len(cities_data)} cities available")
    except Exception as e:
        print(f"  Error loading cities: {e}")
    
    print("="*70)
    print("API URL: http://localhost:5001")
    print("WebSocket: ws://localhost:5001")
    print("Frontend: http://localhost:3000")
    print("="*70)

if __name__ == '__main__':
    print_startup_banner()
    
    # Ensure required directories exist
    output_dir = os.path.join(os.path.dirname(__file__), 'simulation_output')
    os.makedirs(output_dir, exist_ok=True)
    
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')

    print(f"\nStarting Enhanced Edge Computing Simulation API on http://{host}:{port}...")
    print("Press Ctrl+C to stop the server")
    
    try:
        socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        if simulation_manager.current_simulation:
            print("Stopping active simulation...")
            simulation_manager.stop_simulation()
        print("Server stopped.")