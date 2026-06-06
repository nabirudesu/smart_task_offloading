# backend/integrated_simulation.py - Enhanced with trajectory tracking and SimPy sync

import threading
import time
import uuid
import json
import os
import sys
import math
import random
from typing import Dict, List, Any, Optional
from flask_socketio import SocketIO
import ray
# Add the current directory to Python path to import simulation modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config_data import CITY_CONFIGS, UPDATE_FREQUENCY_MS, DEFAULT_SIMULATION_PARAMS

class IntegratedSimulation:
    """Enhanced integrated simulation with trajectory tracking and SimPy synchronization"""
    
    def __init__(self, config: Dict, socketio: SocketIO, simulation_id: str):
        self.config = config
        self.socketio = socketio
        self.simulation_id = simulation_id
        self.is_running = False
        self.start_time = None
        self.real_simulation = None
        self.tracker = None
        
        # Trajectory tracking
        self.vehicle_trajectories = {}  # {vehicle_id: {'planned': [], 'traveled': []}}
        self.edge_server_positions = []
        
        # SimPy time tracking
        self.simpy_start_time = 0
        self.last_simpy_time = 0
        
        # NEW: Handoff tracking
        self.vehicle_handoffs = {}  # {vehicle_id: [{from_edge, to_edge, time}]}
        self.vehicle_last_edge = {}  # {vehicle_id: last_edge_id}

        # Import real simulation components
        self._import_simulation_components()

    def _import_simulation_components(self):
        """Import the real simulation components"""
        try:
            # Import the main simulation class and components
            global EdgeComputingSimulation, Location, initialize_tracker, get_tracker
            from main_simulation import EdgeComputingSimulation
            from entities.location import Location
            from entities.simulation_tracker_exteded import initialize_tracker, get_tracker
            # n print(f"[SIM {self.simulation_id}] Successfully imported real simulation components")
        except ImportError as e:
            # n print(f"[SIM {self.simulation_id}] Failed to import simulation components: {e}")
            raise

    def _create_simulation_with_config(self):
        """Create the real simulation with web interface configuration"""
        try:
            # Create the real simulation instance
            self.real_simulation = EdgeComputingSimulation()
            
            # Override simulation parameters with web config
            self._apply_web_configuration()
            
            # Initialize tracker
            self.tracker = self.real_simulation.tracker
            
            # n print(f"[SIM {self.simulation_id}] Real simulation created with config: {self.config}")
        except Exception as e:
            print(f"[SIM {self.simulation_id}] Failed to create simulation: {e}")
            print("DETAILS")
            raise

    def _apply_web_configuration(self):
        """Apply web interface configuration to the real simulation"""
        if not self.real_simulation:
            return

        self.real_simulation.SCENARIO = self.config.get('scenario', DEFAULT_SIMULATION_PARAMS['scenario'])

        # Update simulation duration
        self.real_simulation.SIMULATION_DURATION = self.config.get('duration', DEFAULT_SIMULATION_PARAMS['duration'])
        
        # Update number of vehicles
        self.real_simulation.NUM_VEHICLES = self.config.get('vehicles', DEFAULT_SIMULATION_PARAMS['vehicles'])
        
        # Update edge coverage radius
        self.real_simulation.EDGE_COVERAGE_RADIUS = self.config.get('edge_coverage', DEFAULT_SIMULATION_PARAMS['edge_coverage'])
        
        # Update vehicle speed
        self.real_simulation.VEHICLE_SPEED = self.config.get('vehicle_speed', DEFAULT_SIMULATION_PARAMS['vehicle_speed'])
        
        # Update vehicle FPS
        self.real_simulation.VEHICLE_FPS = self.config.get('fps', DEFAULT_SIMULATION_PARAMS['fps'])
        
        # Update task generation rate
        self.real_simulation.VEHICLE_TASK_RATE = self.config.get('task_rate', DEFAULT_SIMULATION_PARAMS['task_rate'])
        # Update EDGE servers Number 
        self.real_simulation.NUM_EDGE_SERVERS = self.config.get('edge_servers',1)
        
        # Configure city and positions
        self._configure_city_and_positions()
        
        # n print(f"[SIM {self.simulation_id}] Applied web configuration to real simulation")

    def _configure_city_and_positions(self):
        """Configure city-specific setup and generate positions"""
        # Load cities from Location.json
        cities_path = os.path.join(os.path.dirname(__file__), 'Data', 'locations.json')
        if not os.path.exists(cities_path):
            # n print(f"[WARNING] Location.json not found, using default Paris configuration")
            return
            
        with open(cities_path, 'r') as f:
            cities_data = json.load(f)
        
        # Find selected city
        city_key = self.config.get('city', 'Paris')
        selected_city = None
        for city in cities_data:
            if city['city'].lower().replace(' ', '_') == city_key:
                selected_city = city
                break
        
        if not selected_city:
            # n print(f"[WARNING] City {city_key} not found, using Paris as default")
            selected_city = {"city": "Paris", "latitude": 48.8566, "longitude": 2.3522}
        
        # Update simulation locations
        from entities.location import Location
        
        city_center = [selected_city['latitude'], selected_city['longitude']]
        scenario = self.config.get('scenario', 'general')
        edge_coverage = self.config.get('edge_coverage', 2.0)

        self.real_simulation.EDGE_POSITION = Location(
            selected_city['city'] + " Center",
            selected_city['latitude'],
            selected_city['longitude']
        )
        num_vehicles = self.config.get('vehicles', 5) 
        if scenario == 'mobility':
            self._generate_handoff_scenario(city_center, num_vehicles, edge_coverage)
        else:
            # Generate edge server positions based on scenario
            self._generate_edge_server_positions(city_center,scenario,edge_coverage)

            # Generate destinations for vehicles
            self._generate_destinations(city_center,num_vehicles,scenario,edge_coverage)
            # Generate initial positions for vehicles within edge coverage
            self.vehicle_initial_positions = self._generate_vehicles_position(
                city_center, 
                num_vehicles,
                edge_coverage
            )
            # n print(f"[SIM {self.simulation_id}] Configured simulation for city: {selected_city['city']}")
    
    def _generate_handoff_scenario(self, city_center, num_vehicles, edge_coverage):
        """Generates a specific handoff scenario."""
        from entities.location import Location

        # Speed = 200 km/h, Time = 3s -> Distance = 0.167 km
        distance_km = 0.042  # 42 meters apart for handoff within 3 seconds at 50 km/h
        
        # Create two edge servers
        edge_server_locations = []
        self.edge_server_positions = []

        # First edge server at the city center
        self.edge_server_positions.append({
            'id': 1,
            'position': city_center,
            'coverage': edge_coverage,
            'name': "Edge-1-Handoff-Start"
        })
        edge_server_locations.append(Location(
            city=self.config.get('city', "Paris"),
            latitude=city_center[0],
            longitude=city_center[1]
        ))

        # Second edge server at a calculated offset
        angle = 45  # Place it diagonally for a clear path
        new_position = self._offset_position(city_center, distance_km, angle)
        self.edge_server_positions.append({
            'id': 2,
            'position': new_position,
            'coverage': edge_coverage,
            'name': "Edge-2-Handoff-End"
        })
        edge_server_locations.append(Location(
            city=self.config.get('city', "Paris"),
            latitude=new_position[0],
            longitude=new_position[1]
        ))
        self.real_simulation.EDGE_POSITION = edge_server_locations

        # Set vehicle position and destination
        vehicle_positions = []
        destinations = []

        start_pos = city_center
        end_pos = new_position

        for i in range(num_vehicles):
            # Place vehicle at the center of the first edge server
            vehicle_positions.append(Location(
                f"Start-Vehicle-{i+1}",
                start_pos[0],
                start_pos[1]
            ))
            # Place destination at the center of the second edge server
            destinations.append(Location(
                f"Destination-Vehicle-{i+1}",
                end_pos[0],
                end_pos[1]
            ))

        self.real_simulation.VEHICLE_POSITIONS = vehicle_positions
        self.real_simulation.ROUTE_DESTINATIONS = destinations




    def _generate_edge_server_positions(self, city_center,scenario,edge_coverage):
        """Generate edge server positions based on scenario"""
        # scenario = self.config.get('scenario', 'general')
        # edge_coverage = self.config.get('edge_coverage', 2.0)
        edge_server_locations = []
        self.edge_server_positions = []
        # Multiple edge servers for mobility scenario
        edge_count = self.config.get('edge_servers', 2)
        # Central edge server
        self.edge_server_positions.append({
            'id': 1,
            'position': city_center,
            'coverage': edge_coverage,
            'name': f"Edge-1-Central"
        })
        edge_server_locations.append(Location(
            city=self.config.get('city',"Paris"),
            latitude=city_center[0],
            longitude=city_center[1]
        ))
        
        # Additional edge servers in different directions
        angles = [i * (360 / (edge_count - 1)) for i in range(edge_count - 1)]
        if scenario=="general":
            offset_distance = edge_coverage  # 1.5x coverage distance
        else:
            offset_distance = edge_coverage * 2  # 1.5x coverage distance
        for i, angle in enumerate(angles):
            new_position = self._offset_position(city_center, offset_distance, angle)
            self.edge_server_positions.append({
                'id': i + 2,
                'position': new_position,
                'coverage': edge_coverage,
                'name': f"Edge-{i+2}-{['North', 'East', 'South'][i % 3]}"
            })
            edge_server_locations.append(Location(
                city=self.config.get('city',"Paris"),
                latitude=new_position[0],
                longitude=new_position[1]
            ))        


        self.real_simulation.EDGE_POSITION = edge_server_locations
    
    def _generate_destinations(self, city_center,num_destinations,scenario,edge_coverage):
        """Generate destination points for vehicles"""
        from entities.location import Location

        destinations = []
        
        # Generate destinations in a radius around city center
        for i in range(num_destinations):
            angle = (i * 360 / num_destinations) + random.uniform(-15, 15)  # Add some randomness
            distance = random.uniform(0.2, edge_coverage if scenario =="general" else edge_coverage*2)  # 0.2-zone_linegth km from center
            dest_position = self._offset_position(city_center, distance, angle)
            
            destinations.append(Location(
                f"Destination-{i+1}",
                dest_position[0],
                dest_position[1]
            ))
        
        self.real_simulation.ROUTE_DESTINATIONS = destinations

    def _generate_vehicles_position(self, city_center,num_vehicles,edge_coverage) -> List[Dict]:
            """Generate initial positions for vehicles within the edge coverage zone, each linked to a destination."""
            vehicle_positions = []

            for i in range(num_vehicles):
                # Random distance within edge coverage (0 to coverage radius)
                random_dist = random.uniform(0, edge_coverage)
                # Random angle (0 to 360 degrees)
                random_angle = random.uniform(0, 360)
                
                # Generate initial position
                position = self._offset_position(city_center, random_dist, random_angle)
                
                vehicle_positions.append(Location(
                    f"Destination-{i+1}",
                    position[0],
                    position[1]
                ))
            
            self.real_simulation.VEHICLE_POSITIONS = vehicle_positions

    def _offset_position(self, center, distance_km, angle_degrees):
        """Calculate new position offset from center by distance and angle"""
        lat, lng = center
        
        # Convert distance from km to degrees (approximate)
        lat_offset = (distance_km / 111.0) * math.cos(math.radians(angle_degrees))
        lng_offset = (distance_km / (111.0 * math.cos(math.radians(lat)))) * math.sin(math.radians(angle_degrees))
        
        return [lat + lat_offset, lng + lng_offset]

    def run_simulation(self):
        """Run the real simulation and emit tracker data via WebSocket"""
        try:
            self.is_running = True
            self.socketio.emit('layout_change_trigger', {
                'show_simulation': True,
                'simulation_id': self.simulation_id
            })
            self.start_time = time.time()
            # n print(f"[SIM {self.simulation_id}] Starting real simulation")
            
            # Create and setup the real simulation
            self._create_simulation_with_config()
            print("DEBUG0")
            self.real_simulation.setup_simulation()
            # Initialize trajectory tracking
            print("DEBUG01")
            self._initialize_trajectory_tracking()
            print("DEBUG1")
            
            # Record SimPy start time
            self.simpy_start_time = self.real_simulation.env.now
            
            # Start WebSocket update process in a separate thread
            update_thread = threading.Thread(target=self._websocket_update_loop, daemon=True)
            print("DEBUG2")
            update_thread.start()
            print("DEBUG3")
            
            # Run the real simulation
            self.real_simulation.run_simulation()
            print("DEBUG5")
            # n print("-----------------***************+++++++++++++++++++")
            time.sleep(0.05)
        except Exception as e:
            print(f"[SIM {self.simulation_id}] Simulation error: {e}")
            self.emit_simulation_error(str(e))
        finally:
            self.is_running = False
            self.emit_simulation_ended()

    def _initialize_trajectory_tracking(self):
        """Initialize trajectory tracking for all vehicles"""
        if not self.real_simulation or not self.real_simulation.vehicles:
            return
            
        for vehicle in self.real_simulation.vehicles:
            self.vehicle_trajectories[vehicle.id] = {
                'planned': self._extract_planned_route(vehicle),
                'traveled': [],
                'current_position': vehicle.position.copy() if hasattr(vehicle, 'position') else [0, 0],
                'destination_location': [vehicle.destination_location.latitude, vehicle.destination_location.longitude] if hasattr(vehicle, 'destination_location') else [0, 0]
            }

    def _extract_planned_route(self, vehicle):
        """Extract planned route from vehicle's route_plan"""
        if not hasattr(vehicle, 'route_plan') or not vehicle.route_plan:
            return []
        
        planned_route = []
        for waypoint in vehicle.route_plan:
            if 'coords' in waypoint:
                planned_route.append(list(waypoint['coords']))
        
        return planned_route

    def _websocket_update_loop(self):
        """Send real-time updates using tracker data with SimPy time sync"""
        while self.is_running:
            if self.tracker and self.real_simulation:
                # Update vehicle trajectories
                self._update_vehicle_trajectories()
                
                # Emit update with SimPy time
                self.emit_tracker_update()
                    
                time.sleep(UPDATE_FREQUENCY_MS / 1000.0)
            # except Exception as e:
            #     # n print(f"[SIM {self.simulation_id}] WebSocket update error: {e}")
            #     break

    def _update_vehicle_trajectories(self):
        """Update vehicle trajectory tracking"""
        if not self.real_simulation or not self.real_simulation.vehicles:
            return
            
        for vehicle in self.real_simulation.vehicles:
            if vehicle.id in self.vehicle_trajectories:
                current_pos = vehicle.position.copy() if hasattr(vehicle, 'position') else [0, 0]
                last_pos = self.vehicle_trajectories[vehicle.id]['current_position']
                
                # If position changed, add to traveled path
                if current_pos != last_pos:
                    self.vehicle_trajectories[vehicle.id]['traveled'].append(current_pos.copy())
                    self.vehicle_trajectories[vehicle.id]['current_position'] = current_pos

    def emit_tracker_update(self):
        """Emit simulation state using real tracker data with SimPy time"""
        try:
            # Get SimPy time
            simpy_time = self.real_simulation.env.now if self.real_simulation else 0
            real_time = time.time() - self.start_time if self.start_time else 0
            
            # Get data from tracker snapshots
            vehicles_data = self._extract_vehicles_data()
            edge_servers_data = self._extract_edge_servers_data()
            cloud_server_data = self._extract_cloud_server_data()
            tasks_data = self._extract_tasks_data()
            
            update_data = {
                'timestamp': real_time,
                'simpy_time': simpy_time,
                'simulation_id': self.simulation_id,
                'vehicles': vehicles_data,
                'edge_servers': edge_servers_data,
                'cloud_server': cloud_server_data,
                'tasks': tasks_data,
                'scenario': self.config['scenario'],
                'city': self.config['city'],
                'duration': self.config.get('duration', 30),
                'vehicle_trajectories': self.vehicle_trajectories,
                'stats': self._get_current_stats()
            }
            
            self.socketio.emit('simulation_update', update_data)
            
            # Check if simulation should complete based on duration
            duration = self.config.get('duration', 30)
            if simpy_time >= duration:
                # n print(f"[SIM {self.simulation_id}] Simulation duration reached: {simpy_time}/{duration}")
                self.is_running = False
                
        except Exception as e:
            print(f"[SIM {self.simulation_id}] Error emitting tracker update: {e}")

    def _extract_vehicles_data(self) -> List[Dict]:
        """Extract vehicle data from tracker snapshots with trajectory info"""
        vehicles_data = []
        
        if not self.tracker or not self.tracker.server_snapshots:
            return vehicles_data
            
        for server_id, snapshot in self.tracker.server_snapshots.items():
            if snapshot.server_type == 'vehicle':
                vehicle_data = {
                    'id': snapshot.server_id,
                    'name': snapshot.server_name,
                    'position': snapshot.position if hasattr(snapshot, 'position') else [0, 0],
                    'speed': snapshot.speed if hasattr(snapshot, 'speed') else 0,
                    'trip_finished': snapshot.trip_finished if hasattr(snapshot, 'trip_finished') else False,
                    'power_of_vehicle': snapshot.power_of_vehicle if hasattr(snapshot, 'power_of_vehicle') else 0,
                    'current_edge_server': 1,
                    'ram_usage': (snapshot.ram_capacity - snapshot.available_ram) / snapshot.ram_capacity if snapshot.ram_capacity > 0 else 0,
                    'platform_states': snapshot.platform_states,
                    # Add trajectory info
                    'trajectory': self.vehicle_trajectories.get(snapshot.server_id, {
                        'planned': snapshot.route_plan,
                        'traveled': [],
                        'current_position': [0, 0]
                    }),
                    'handoffs':self.tracker.handoff_snapshots.get(snapshot.server_id, [])
                }
                
                # Add task statistics for this vehicle
                vehicle_tasks = self._get_vehicle_tasks(snapshot.server_id)
                vehicle_data.update(vehicle_tasks)
                
                vehicles_data.append(vehicle_data)
                
        return vehicles_data

    def _extract_edge_servers_data(self) -> List[Dict]:
        """Extract edge server data with configured positions"""
        edge_servers_data = []
        
        if not self.real_simulation.tracker or not self.real_simulation.tracker.server_snapshots:
            return edge_servers_data
            
        for server_id, snapshot in self.real_simulation.tracker.server_snapshots.items():
            if snapshot.server_type == 'edge':
                # Find corresponding position from our generated positions
                position = [0, 0]
                coverage = self.config.get('edge_coverage', 2.0)
                
                for edge_pos in self.edge_server_positions:
                    if edge_pos['position'] == snapshot.position:
                        position = edge_pos['position']
                        coverage = edge_pos['coverage']
                        break
                
                edge_data = {
                    'id': snapshot.server_id,
                    'name': snapshot.server_name,
                    'position': position,
                    'coverage_radius': coverage,
                    'queue_size': snapshot.queue_size if hasattr(snapshot, 'queue_size') else 0,
                    'queue_capacity': snapshot.queue_capacity if hasattr(snapshot, 'queue_capacity') else 10,
                    'coverage_length': snapshot.coverage_length if hasattr(snapshot, 'coverage_length') else 2000,
                    'ram_usage': (snapshot.ram_capacity - snapshot.available_ram) / snapshot.ram_capacity if snapshot.ram_capacity > 0 else 0,
                    'platform_states': snapshot.platform_states,
                    'connected_vehicles': []
                }
                
                # Add task statistics for this edge server
                edge_tasks = self._get_server_tasks(snapshot.server_id, 'edge')
                edge_data.update(edge_tasks)
                
                edge_servers_data.append(edge_data)
        return edge_servers_data

    def _extract_cloud_server_data(self) -> Dict:
        """Extract cloud server data from tracker snapshots"""
        if not self.tracker or not self.tracker.server_snapshots:
            return {'id': 'cloud-1', 'active_tasks': 0, 'total_processed': 0}
            
        for server_id, snapshot in self.tracker.server_snapshots.items():
            if snapshot.server_type == 'cloud':
                cloud_data = {
                    'id': snapshot.server_id,
                    'name': snapshot.server_name,
                    'position': [0, 0],  # Cloud has no geographic position
                    'ram_usage': (snapshot.ram_capacity - snapshot.available_ram) / snapshot.ram_capacity if snapshot.ram_capacity > 0 else 0,
                    'bandwidth': snapshot.bandwidth,
                    'platform_states': snapshot.platform_states
                }
                
                # Add task statistics for cloud server
                cloud_tasks = self._get_server_tasks(snapshot.server_id, 'cloud')
                cloud_data.update(cloud_tasks)
                
                return cloud_data
                
        return {'id': 'cloud-1', 'active_tasks': 0, 'total_processed': 0}

    def _extract_tasks_data(self) -> List[Dict]:
        """Extract task data from tracker snapshots"""
        tasks_data = []
        if not self.tracker or not self.tracker.task_snapshots:
            return tasks_data
            
        for task_id, task_snapshot in self.tracker.task_snapshots.items():
            task_data = {
                'id': task_snapshot.task_id,
                'type': task_snapshot.task_type,
                'status': task_snapshot.status,
                'vehicle_id': task_snapshot.vehicle_id,
                'min_accuracy': task_snapshot.min_accuracy,
                'max_latency': task_snapshot.max_latency,
                'data_size': task_snapshot.data_size,
                'arrival_time': task_snapshot.arrival_time,
                'execution_start_time': task_snapshot.execution_start_time,
                'execution_end_time': task_snapshot.execution_end_time,
                'response_time': task_snapshot.execution_end_time - task_snapshot.arrival_time if task_snapshot.execution_end_time > task_snapshot.arrival_time and task_snapshot.execution_end_time>0 else 0,
                'chosen_execution_memory_consumption': task_snapshot.chosen_execution_memory_consumption,
                'chosen_execution_execution_time': task_snapshot.chosen_execution_execution_time,
                'chosen_execution_energy_consumption': task_snapshot.chosen_execution_energy_consumption,
                'chosen_execution_platform_usage' : task_snapshot.chosen_execution_platform_usage,
                'chosen_execution_model_accuracy': task_snapshot.chosen_execution_model_accuracy,
                'chosen_execution_level': task_snapshot.chosen_execution_level,
                'chosen_execution_platform': task_snapshot.chosen_execution_platform,
                'chosen_execution_model': task_snapshot.chosen_execution_model
            }
            
            tasks_data.append(task_data)
            
        return tasks_data

    def _get_vehicle_tasks(self, vehicle_id: int) -> Dict:
        """Get task statistics for a specific vehicle"""
        if not self.tracker or not self.tracker.task_snapshots:
            return {
                'total_tasks': 0,
                'tasks_completed': 0,
                'tasks_failed': 0,
                'current_task_count': 0
            }
            
        vehicle_tasks = [task for task in self.tracker.task_snapshots.values()
                        if task.vehicle_id == vehicle_id]
        
        completed = len([t for t in vehicle_tasks if t.status == 'SUCCESS'])
        failed = len([t for t in vehicle_tasks if t.status in ['FAILED', 'FAILED_RESOURCE_UNAVAILABLE']])
        active = len([t for t in vehicle_tasks if t.status in ['CREATED', 'IN_EXEC']])
        
        return {
            'total_tasks': len(vehicle_tasks),
            'tasks_completed': completed,
            'tasks_failed': failed,
            'current_task_count': active
        }

    def _get_server_tasks(self, server_id: int, server_type: str) -> Dict:
        """Get task statistics for a specific server"""
        return {
            'active_tasks': 0,
            'total_processed': 0
        }

    def _get_current_stats(self) -> Dict:
        """Get current simulation statistics from tracker"""
        if not self.tracker:
            return {}
        return self.tracker.get_summary_statistics()

    def emit_simulation_ended(self):
        """Emit simulation ended event with final statistics"""
        end_data = {
            'simulation_id': self.simulation_id,
            'message': 'Simulation completed',
            'final_stats': self._get_final_stats(),
            'simpy_time': self.real_simulation.env.now if self.real_simulation else 0,
            'timestamp': time.time() - self.start_time if self.start_time else 0
        }
        
        self.socketio.emit('simulation_ended', end_data)
        # n print(f"[SIM {self.simulation_id}] Simulation ended, final stats emitted")
        ray.shutdown()

    def emit_simulation_error(self, error_message: str):
        """Emit simulation error event"""
        error_data = {
            'simulation_id': self.simulation_id,
            'message': 'Simulation error occurred',
            'error': error_message
        }
        
        self.socketio.emit('simulation_error', error_data)
        # n print(f"[SIM {self.simulation_id}] Simulation error emitted: {error_message}")

    def _get_final_stats(self) -> Dict:
        """Get final simulation statistics"""
        if not self.tracker:
            return {'total_tasks_generated': 0, 'total_tasks_completed': 0}
            
        stats = self.tracker.get_summary_statistics()
        return {
            'total_tasks_generated': stats.get('total_tasks_generated', 0),
            'total_tasks_completed': stats.get('tasks_completed', 0),
            'total_tasks_failed': stats.get('tasks_failed', 0),
            'success_rate': stats.get('success_rate', 0),
            'simulation_duration': stats.get('simulation_duration', 0),
            'total_events': stats.get('total_events', 0)
        }

    def stop(self):
        """Stop the real simulation"""
        self.is_running = False
        if self.real_simulation:
            if hasattr(self.real_simulation, 'stop'):
                self.real_simulation.stop()
        # n print(f"[SIM {self.simulation_id}] Simulation stop requested")


class SimulationManager:
    """Enhanced simulation manager with trajectory tracking"""
    
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.current_simulation: Optional[IntegratedSimulation] = None
        self.simulation_thread: Optional[threading.Thread] = None
        self.simulation_id: Optional[str] = None

    def start_simulation(self, config: Dict) -> Dict:
        """Start a new real simulation with given configuration"""
        try:
            # Stop current simulation if running
            if self.current_simulation and self.current_simulation.is_running:
                self.stop_simulation()
                
            # Validate configuration
            validation_result = self._validate_config(config)
            print(validation_result)
            if not validation_result['valid']:
                return {
                    "success": False,
                    "error": f"Invalid configuration: {validation_result['error']}"
                }
                
            # Generate unique simulation ID
            self.simulation_id = str(uuid.uuid4())[:8]
            
            # Create new integrated simulation instance
            self.current_simulation = IntegratedSimulation(config, self.socketio, self.simulation_id)
            print("DEBUGGING")
            # Start simulation in separate thread
            self.simulation_thread = threading.Thread(
                target=self.current_simulation.run_simulation(),
                daemon=True
            )
            self.simulation_thread.start()
            
            # n print(f"[MANAGER] Started enhanced simulation {self.simulation_id} with config: {config}")
            
            return {
                "success": True,
                "simulation_id": self.simulation_id,
                "message": "Enhanced simulation started successfully"
            }
            
        except Exception as e:
            print(f"[MANAGER] Failed to start simulation: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _validate_config(self, config: Dict) -> Dict:
        """Validate simulation configuration"""
        required_fields = ['scenario', 'vehicles', 'duration', 'city']
        for field in required_fields:
            if field not in config:
                return {'valid': False, 'error': f"Missing required field: {field}"}
                
        # Validate ranges
        if not (1 <= config['vehicles'] <= 30):
            return {'valid': False, 'error': "Vehicles must be between 1 and 30"}
            
        if not (1 <= config['duration'] <=5):
            return {'valid': False, 'error': "Duration must be between 1 and 5 seconds"}
            
        # Validate edge server count for mobility scenario
        if config.get('scenario') == 'mobility':
            edge_servers = config.get('edge_servers', 2)
            if not (1 <= edge_servers <= 5):
                return {'valid': False, 'error': "Edge servers must be between 2 and 4 for mobility scenario"}
                
        return {'valid': True}

    def stop_simulation(self) -> Dict:
        """Stop the current real simulation"""
        try:
            if self.current_simulation:
                self.current_simulation.stop()
                if self.simulation_thread and self.simulation_thread.is_alive():
                    self.simulation_thread.join(timeout=5)
                    
            # n print(f"[MANAGER] Stopped simulation {self.simulation_id}")
            
            return {
                "success": True,
                "message": "Enhanced simulation stopped successfully"
            }
            
        except Exception as e:
            # n print(f"[MANAGER] Failed to stop simulation: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_status(self) -> Dict:
        """Get current simulation status"""
        if not self.current_simulation:
            return {
                "running": False,
                "simulation_id": None,
                "message": "No simulation running"
            }
            
        return {
            "running": self.current_simulation.is_running,
            "simulation_id": self.simulation_id,
            "config": self.current_simulation.config,
            "uptime": time.time() - self.current_simulation.start_time if self.current_simulation.start_time else 0
        }