"""Configuration settings for the Edge Computing Simulation"""

# Load city data from JSON
import json
import os

# Predefined configuration options for frontend dropdowns
SIMULATION_CONFIGS = {
    "scenarios": [
        {"value": "general", "label": "General Scenario"},
        {"value": "mobility", "label": "Mobility Test Scenario"}
    ],
    "vehicles": [
        {"value": 1, "label": "1 Vehicle"},
        {"value": 3, "label": "3 Vehicles"}, 
        {"value": 5, "label": "5 Vehicles"},
        {"value": 10, "label": "10 Vehicles"},
        {"value": 15, "label": "15 Vehicles"},
        {"value": 20, "label": "20 Vehicles"},
        {"value": 30, "label": "30 Vehicles"}
    ],
    "duration": [
        {"value": 1, "label": "1 seconds"},
        {"value": 2, "label": "2 seconds"},
        {"value": 3, "label": "3 seconds"},
        {"value": 4, "label": "4 seconds"},
        {"value": 5, "label": "5 seconds"}
    ],
    "task_rate": [
        {"value": 1, "label": "1 task/Frame"},
        {"value": 2, "label": "2 tasks/Frame"},
        {"value": 3, "label": "3 tasks/Frame"},
        {"value": 4, "label": "4 tasks/Frame"},
        {"value": 5, "label": "5 tasks/Frame"}
    ],
    "fps": [
        {"value": 1, "label": "1 FPS"},
        {"value": 2, "label": "2 FPS"},
        {"value": 5, "label": "5 FPS"},
        {"value": 10, "label": "10 FPS"},
        {"value": 15, "label": "15 FPS"},
        {"value": 20, "label": "20 FPS"}
    ],
    "vehicle_speed": [
        {"value": 30, "label": "30 km/h (City)"},
        {"value": 50, "label": "50 km/h (Urban)"},
        {"value": 80, "label": "80 km/h (Highway)"},
        {"value": 120, "label": "120 km/h (Highway)"},
        {"value": 200, "label": "200 km/h (Highway)"}
    ],
    "edge_coverage": [
        {"value": 0.05, "label": "50 m radius"},
        {"value": 0.1, "label": "100 m radius"},
        {"value": 0.2, "label": "200 m radius"},
        {"value": 0.5, "label": "500 m radius"},
        {"value": 1.0, "label": "1 km radius"},
        {"value": 1.5, "label": "1.5 km radius"},
        {"value": 2.0, "label": "2 km radius"},
        {"value": 3.0, "label": "3 km radius"},
        {"value": 5.0, "label": "5 km radius"}
    ]
}

# Load cities from data file
def load_cities():
    """Load cities from the JSON data file"""
    try:
        data_file = os.path.join(os.path.dirname(__file__), 'Data/locations.json')
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        cities_data = data  # Cities are in element 1
        cities = []
        
        for city_data in cities_data:
            cities.append({
                "value": city_data["city"].lower().replace(" ", "_"),
                "label": f"{city_data['city']}",
                "center": [city_data["latitude"], city_data["longitude"]]
            })
        
        return cities
    except Exception as e:
        print(f"Error loading cities: {e}")
        # Fallback to Paris if loading fails
        return [{"value": "paris", "label": "Paris, France", "center": [48.8566, 2.3522]}]

# Add cities to configuration
SIMULATION_CONFIGS["cities"] = load_cities()

# City-specific configurations
CITY_CONFIGS = {}

def generate_city_configs():
    """Generate city configurations based on loaded city data"""
    global CITY_CONFIGS
    
    try:
        data_file = os.path.join(os.path.dirname(__file__), 'Data/locations.json')
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        cities_data = data  # Cities are in element 1
        
        for city_data in cities_data:
            city_key = city_data["city"].lower().replace(" ", "_")
            city_name = city_data["city"]
            lat = city_data["latitude"]
            lng = city_data["longitude"]
            
            # Generate destinations in a radius around the city center
            destinations = []
            for i in range(5):
                # Create destinations in different directions from center
                angle = (i * 72) * 3.14159 / 180  # 72 degrees apart
                radius = 0.01  # ~1km radius
                dest_lat = lat + radius * np.cos(angle) if 'np' in globals() else lat + 0.005
                dest_lng = lng + radius * np.sin(angle) if 'np' in globals() else lng + 0.005
                
                destinations.append({
                    "lat": dest_lat,
                    "lng": dest_lng, 
                    "name": f"{city_name} Destination {i+1}"
                })
            
            CITY_CONFIGS[city_key] = {
                "center": {"lat": lat, "lng": lng, "name": f"{city_name} Center"},
                "destinations": destinations,
                "edge_servers": {
                    "general": [
                        {"lat": lat, "lng": lng, "name": f"{city_name} Central Edge", "coverage": 2.0}
                    ],
                    "mobility": [
                        {"lat": lat, "lng": lng, "name": f"{city_name} Central Edge", "coverage": 1.5},
                        {"lat": lat + 0.005, "lng": lng + 0.005, "name": f"{city_name} North Edge", "coverage": 1.5},
                        {"lat": lat - 0.005, "lng": lng - 0.005, "name": f"{city_name} South Edge", "coverage": 1.5}
                    ]
                }
            }
    except Exception as e:
        print(f"Error generating city configs: {e}")
        # Fallback configuration
        CITY_CONFIGS["paris"] = {
            "center": {"lat": 48.8566, "lng": 2.3522, "name": "Paris Center"},
            "destinations": [
                {"lat": 48.8606, "lng": 2.3376, "name": "Louvre"},
                {"lat": 48.8530, "lng": 2.3499, "name": "Notre Dame"}
            ],
            "edge_servers": {
                "general": [{"lat": 48.8566, "lng": 2.3522, "name": "Paris Central Edge", "coverage": 2.0}],
                "mobility": [{"lat": 48.8566, "lng": 2.3522, "name": "Paris Central Edge", "coverage": 1.5}]
            }
        }

# Import numpy for calculations if available
try:
    import numpy as np
    generate_city_configs()
except ImportError:
    # Generate without numpy
    generate_city_configs()

# WebSocket update frequency (milliseconds)  
UPDATE_FREQUENCY_MS = 10

# Default simulation parameters
DEFAULT_SIMULATION_PARAMS = {
    "duration": 1.0,          # seconds
    "vehicles": 5,             # number of vehicles
    "task_rate": 3,           # tasks per second
    "fps": 1,                  # frames per second
    "vehicle_speed": 50,       # km/h
    "edge_coverage": 1.0,      # km radius
    "scenario": "general",     # simulation scenario
    "city": "paris"           # default city
}

# Task types available in simulation
TASK_TYPES = {
    "DO": "Object Detection",
    "CI": "Classification", 
    "S": "Segmentation",
    "OT": "Object Tracking",
    "TLD": "Traffic Light Detection"
}