# # backend/simulation_wrapper.py
# """
# Wrapper class to integrate the existing simulation with Flask and WebSocket
# """
# import threading
# import time
# import uuid
# import simpy
# import numpy as np
# from typing import Dict, List, Any, Optional
# from flask_socketio import SocketIO

# # Import your existing simulation classes
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from config_data import CITY_CONFIGS, UPDATE_FREQUENCY_MS, SCENARIO_DURATIONS

# class Location:
#     """Simplified location class for the wrapper"""
#     idx = 0
    
#     def __init__(self, city: str, latitude: float, longitude: float):
#         Location.idx += 1
#         self.id = Location.idx
#         self.city = city
#         self.latitude = latitude
#         self.longitude = longitude

# class WebSimulation:
#     """Modified simulation class that emits data via WebSocket"""
    
#     def __init__(self, config: Dict, socketio: SocketIO, simulation_id: str):
#         self.config = config
#         self.socketio = socketio
#         self.simulation_id = simulation_id
#         self.env = simpy.Environment()
#         self.vehicles: List = []
#         self.edge_servers: List = []
#         self.cloud_server = None
#         self.is_running = False
#         self.start_time = None
        
#         # City configuration
#         self.city_config = CITY_CONFIGS[config['city']]
        
#         # Mock simulation data for now (we'll integrate your full simulation later)
#         self.setup_mock_simulation()
    
#     def setup_mock_simulation(self):
#         """Setup mock simulation data - will be replaced with your full simulation"""
        
#         # Create mock vehicles
#         for i in range(self.config['vehicles']):
#             center = self.city_config['center']
#             destinations = self.city_config['destinations']
            
#             vehicle = {
#                 'id': i + 1,
#                 'name': f'Vehicle-{i + 1}',
#                 'position': [center['lat'], center['lng']],
#                 'destination': destinations[i % len(destinations)],
#                 'route': [],
#                 'current_task_count': 0,
#                 'total_tasks': 0,
#                 'tasks_completed': 0,
#                 'tasks_failed': 0,
#                 'current_edge_server': 1,
#                 'trip_finished': False
#             }
#             self.vehicles.append(vehicle)
        
#         # Create edge servers based on scenario
#         scenario = self.config['scenario']
#         edge_configs = self.city_config['edge_servers'][scenario]
        
#         for i, edge_config in enumerate(edge_configs):
#             edge_server = {
#                 'id': i + 1,
#                 'name': edge_config['name'],
#                 'position': [edge_config['lat'], edge_config['lng']],
#                 'coverage_radius': edge_config['coverage'],
#                 'queue_size': 0,
#                 'queue_capacity': 10,
#                 'active_tasks': 0,
#                 'total_processed': 0,
#                 'connected_vehicles': []
#             }
#             self.edge_servers.append(edge_server)
        
#         # Create mock cloud server
#         self.cloud_server = {
#             'id': 'cloud-1',
#             'name': 'Cloud Server',
#             'position': [0, 0],  # Cloud has no physical position
#             'active_tasks': 0,
#             'total_processed': 0,
#             'ram_usage': 0.3,
#             'cpu_usage': 0.25
#         }
    
#     def run_simulation(self):
#         """Run the simulation and emit updates via WebSocket"""
#         self.is_running = True
#         self.start_time = time.time()
        
#         print(f"[SIM {self.simulation_id}] Starting simulation with config: {self.config}")
        
#         # Start simulation processes
#         self.env.process(self.vehicle_movement_process())
#         self.env.process(self.task_generation_process())
#         self.env.process(self.websocket_update_process())
        
#         # Run simulation
#         duration = SCENARIO_DURATIONS[self.config['scenario']]
#         try:
#             self.env.run(until=duration)
#         except Exception as e:
#             print(f"[SIM {self.simulation_id}] Simulation error: {e}")
#         finally:
#             self.is_running = False
#             self.emit_simulation_ended()
    
#     def vehicle_movement_process(self):
#         """Simulate vehicle movement"""
#         while self.is_running:
#             for vehicle in self.vehicles:
#                 if not vehicle['trip_finished']:
#                     # Simple movement simulation
#                     dest = vehicle['destination']
#                     current_pos = vehicle['position']
                    
#                     # Move towards destination
#                     lat_diff = dest['lat'] - current_pos[0]
#                     lng_diff = dest['lng'] - current_pos[1]
                    
#                     # Speed factor based on FPS and vehicle speed (50 km/h)
#                     speed_factor = 0.0001 * (self.config['fps'] / 10)
                    
#                     new_lat = current_pos[0] + lat_diff * speed_factor
#                     new_lng = current_pos[1] + lng_diff * speed_factor
                    
#                     vehicle['position'] = [new_lat, new_lng]
                    
#                     # Check if reached destination
#                     distance = abs(lat_diff) + abs(lng_diff)
#                     if distance < 0.001:  # Close enough to destination
#                         vehicle['trip_finished'] = True
            
#             yield self.env.timeout(0.1)  # Update every 100ms
    
#     def task_generation_process(self):
#         """Simulate task generation"""
#         while self.is_running:
#             for vehicle in self.vehicles:
#                 if not vehicle['trip_finished']:
#                     # Generate tasks based on tasks_per_frame and fps
#                     tasks_to_generate = np.random.poisson(self.config['tasks_per_frame'])
                    
#                     vehicle['current_task_count'] += tasks_to_generate
#                     vehicle['total_tasks'] += tasks_to_generate
                    
#                     # Simulate task completion
#                     if vehicle['current_task_count'] > 0:
#                         completed = min(vehicle['current_task_count'], np.random.poisson(2))
#                         vehicle['current_task_count'] -= completed
#                         vehicle['tasks_completed'] += completed
            
#             # Update edge server queues
#             for edge in self.edge_servers:
#                 # Simulate queue changes
#                 edge['queue_size'] = max(0, edge['queue_size'] + np.random.randint(-2, 3))
#                 edge['queue_size'] = min(edge['queue_size'], edge['queue_capacity'])
#                 edge['active_tasks'] = np.random.randint(0, 5)
            
#             yield self.env.timeout(1.0 / self.config['fps'])
    
#     def websocket_update_process(self):
#         """Send updates via WebSocket every UPDATE_FREQUENCY_MS"""
#         while self.is_running:
#             try:
#                 self.emit_simulation_update()
#                 yield self.env.timeout(UPDATE_FREQUENCY_MS / 1000.0)  # Convert ms to seconds
#             except Exception as e:
#                 print(f"[SIM {self.simulation_id}] WebSocket update error: {e}")
#                 break
    
#     def emit_simulation_update(self):
#         """Emit current simulation state via WebSocket"""
#         current_time = time.time() - self.start_time if self.start_time else 0
        
#         update_data = {
#             'timestamp': current_time,
#             'simulation_id': self.simulation_id,
#             'vehicles': self.vehicles,
#             'edge_servers': self.edge_servers,
#             'cloud_server': self.cloud_server,
#             'scenario': self.config['scenario'],
#             'city': self.config['city']
#         }
#         print("Simulation data:" ,update_data)
#         self.socketio.emit('simulation_update', update_data)
    
#     def emit_simulation_ended(self):
#         """Emit simulation ended event"""
#         end_data = {
#             'simulation_id': self.simulation_id,
#             'message': 'Simulation completed',
#             'final_stats': self.get_final_stats()
#         }
#         self.socketio.emit('simulation_ended', end_data)
    
#     def get_final_stats(self):
#         """Get final simulation statistics"""
#         total_tasks = sum(v['total_tasks'] for v in self.vehicles)
#         total_completed = sum(v['tasks_completed'] for v in self.vehicles)
#         total_failed = sum(v['tasks_failed'] for v in self.vehicles)
        
#         return {
#             'total_tasks_generated': total_tasks,
#             'total_tasks_completed': total_completed,
#             'total_tasks_failed': total_failed,
#             'success_rate': total_completed / total_tasks if total_tasks > 0 else 0,
#             'vehicles_completed': sum(1 for v in self.vehicles if v['trip_finished'])
#         }
    
#     def stop(self):
#         """Stop the simulation"""
#         self.is_running = False

# class SimulationManager:
#     """Manages simulation instances"""
    
#     def __init__(self, socketio: SocketIO):
#         self.socketio = socketio
#         self.current_simulation: Optional[WebSimulation] = None
#         self.simulation_thread: Optional[threading.Thread] = None
#         self.simulation_id: Optional[str] = None
    
#     def start_simulation(self, config: Dict) -> Dict:
#         """Start a new simulation with given configuration"""
#         try:
#             # Stop current simulation if running
#             if self.current_simulation and self.current_simulation.is_running:
#                 self.stop_simulation()
            
#             # Generate unique simulation ID
#             self.simulation_id = str(uuid.uuid4())[:8]
            
#             # Create new simulation instance
#             self.current_simulation = WebSimulation(config, self.socketio, self.simulation_id)
            
#             # Start simulation in separate thread
#             self.simulation_thread = threading.Thread(
#                 target=self.current_simulation.run_simulation,
#                 daemon=True
#             )
#             self.simulation_thread.start()
            
#             return {
#                 "success": True,
#                 "simulation_id": self.simulation_id,
#                 "message": "Simulation started successfully"
#             }
            
#         except Exception as e:
#             return {
#                 "success": False,
#                 "error": str(e)
#             }
    
#     def stop_simulation(self) -> Dict:
#         """Stop the current simulation"""
#         try:
#             if self.current_simulation:
#                 self.current_simulation.stop()
                
#             if self.simulation_thread and self.simulation_thread.is_alive():
#                 self.simulation_thread.join(timeout=2)
            
#             return {
#                 "success": True,
#                 "message": "Simulation stopped successfully"
#             }
            
#         except Exception as e:
#             return {
#                 "success": False,
#                 "error": str(e)
#             }
    
#     def get_status(self) -> Dict:
#         """Get current simulation status"""
#         if not self.current_simulation:
#             return {
#                 "running": False,
#                 "simulation_id": None,
#                 "message": "No simulation running"
#             }
        
#         return {
#             "running": self.current_simulation.is_running,
#             "simulation_id": self.simulation_id,
#             "config": self.current_simulation.config,
#             "uptime": time.time() - self.current_simulation.start_time if self.current_simulation.start_time else 0
#         }