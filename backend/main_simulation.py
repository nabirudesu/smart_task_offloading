# main_simulation.py

import os
import simpy

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BACKEND_DIR, "Data")
import numpy as np
import json
import time
from typing import List, Dict, Any, Optional
import ray
from entities.location import Location
from entities.network import Network
from entities.servers.vehicle_server import (
    VehicleServer,
    load_dnn_models,
    load_processing_platforms,
    Level,
)
from entities.servers.edge_server import EdgeServer
from entities.servers.cloud_server import CloudServer
from entities.task import Task, TaskStatus
from entities.simulation_tracker_exteded import initialize_tracker, get_tracker, EventType

# from entities.tracker import Tracker

# Simulation Configuration
SIMULATION_DURATION = 3.0  # seconds
NUM_VEHICLES = 5
EDGE_COVERAGE_RADIUS = 2.0  # km
VEHICLE_SPEED = 50  # km/h
VEHICLE_FPS = 1  # frames per second

# Paris city center coordinates for short routes
PARIS_CENTER = Location("Paris", 48.8566, 2.3522)
ROUTE_DESTINATIONS = [
    Location("Louvre", 48.8606, 2.3376),
    Location("Notre Dame", 48.8530, 2.3499),
    Location("Place de la Bastille", 48.8532, 2.3699),
    Location("Pantheon", 48.8462, 2.3464),
    Location("Saint-Germain", 48.8542, 2.3370),
]


class EdgeComputingSimulation:
    """Main simulation orchestrator for edge computing task offloading scenario."""

    def __init__(self):
        self.SCENARIO = "general" 
        self.SIMULATION_DURATION = SIMULATION_DURATION # seconds
        self.NUM_VEHICLES = NUM_VEHICLES
        self.EDGE_COVERAGE_RADIUS = EDGE_COVERAGE_RADIUS  # km
        self.VEHICLE_SPEED = VEHICLE_SPEED # km/h
        self.VEHICLE_FPS = VEHICLE_FPS  # frames per second
        self.VEHICLE_TASK_RATE = 5
        self.NUM_EDGE_SERVERS:int
        self.EDGE_POSITION:list[Location]
        self.ROUTE_DESTINATIONS:list[Location]
        self.VEHICLE_POSITIONS:list[Location]
        self.env = simpy.Environment()
        self.network = Network(self.env)  # Empty constructor as requested
        self.vehicles: List[VehicleServer] = []
        self.edge_servers: List[EdgeServer] = []
        self.cloud_server: Optional[CloudServer] = None
        self.global_task_details = self._load_task_details()
        # Initialize tracking
        output_dir = os.path.join(os.path.dirname(__file__), "simulation_output")
        self.tracker = initialize_tracker(output_dir)

    def _load_task_details(self) -> Dict[str, Dict]:
        """Load task type definitions for the simulation."""
        return {
            "DO": {  # Object Detection
                "type": "DO",
                "min_accuracy": 0.7,
                "max_latency": 0.5,
                "data_size": 2.0,
                "data_size_output": 0.5,
                "status": TaskStatus.CREATED,
                "arrival_time": 0.0,
            },
            "CI": {  # Classification
                "type": "CI",
                "min_accuracy": 0.8,
                "max_latency": 0.2,
                "data_size": 1.0,
                "data_size_output": 0.1,
                "status": TaskStatus.CREATED,
                "arrival_time": 0.0,
            },
            "S": {  # Segmentation
                "type": "S",
                "min_accuracy": 0.75,
                "max_latency": 0.8,
                "data_size": 3.0,
                "data_size_output": 1.0,
                "status": TaskStatus.CREATED,
                "arrival_time": 0.0,
            },
            "OT": {  # Object Tracking
                "type": "OT",
                "min_accuracy": 0.7,
                "max_latency": 0.5,
                "data_size": 2.5,
                "data_size_output": 0.3,
                "status": TaskStatus.CREATED,
                "arrival_time": 0.0,
            },
            "TLD": {  # Traffic Light Detection
                "type": "TLD",
                "min_accuracy": 0.85,
                "max_latency": 0.3,
                "data_size": 1.5,
                "data_size_output": 0.2,
                "status": TaskStatus.CREATED,
                "arrival_time": 0.0,
            },
        }

    def setup_cloud_server(self) -> CloudServer:
        """Initialize the cloud server with processing platforms and models."""
        print("[SETUP] Creating cloud server...")

        # Load cloud processing platforms
        cloud_platforms = load_processing_platforms(os.path.join(_DATA_DIR, "processing_platforms.json"), "C", self.env)

        # Load cloud DNN models
        cloud_models = load_dnn_models(os.path.join(_DATA_DIR, "dnn_models.json"), "C")

        # Create cloud server
        cloud_server = CloudServer(
            level=Level.CLOUD,
            location=Location("Cloud Data Center", 48.8566, 2.3522),
            processing_platforms_list=cloud_platforms,
            deployed_models=cloud_models,
            ram_capacity=128.0,  # GB
            avilable_ram=120.0,  # GB
            bandwidth=10000.0,  # Mbps
            power_P_c_e =20.0,
            # tracker = self.tracker,
            env=self.env,
        )
        cloud_server.setPPList(cloud_platforms)
        cloud_server.setModelList(cloud_models)
        # Register server with tracker
        self.tracker.register_server(cloud_server)
        print(
            f"[SETUP] Cloud server '{cloud_server.name}' created with {len(cloud_platforms)} platforms and {len(cloud_models)} models"
        )
        return cloud_server

    def setup_edge_servers(self) -> List[EdgeServer]:
        """Initialize edge servers along the route."""
        print("[SETUP] Creating edge servers...")

        edge_servers = []

        # Create edge server at Paris center
        edge_platforms = load_processing_platforms(os.path.join(_DATA_DIR, "processing_platforms.json"), "E", self.env)
        edge_models = load_dnn_models(os.path.join(_DATA_DIR, "dnn_models.json"), "E")
        print("number of edge servers",self.NUM_EDGE_SERVERS)
        print("number of edge positions",len(self.EDGE_POSITION))
        for i in range(self.NUM_EDGE_SERVERS):
            edge_server = EdgeServer(
                level=Level.EDGE,
                location=self.EDGE_POSITION[i],
                length=self.EDGE_COVERAGE_RADIUS * 1000,  # Convert km to meters
                processing_platforms_list=edge_platforms,
                deployed_models=edge_models,
                ram_capacity=32.0,  # GB
                avilable_ram=28.0,  # GB
                bandwidth=1000.0,  # Mbps
                power_P_e_e=50.0,  # Watts - edge to edge communication
                power_P_e_v=30.0,  # Watts - edge to vehicle communication
                power_P_e_c=80.0,  # Watts - edge to cloud communication
                network=self.network,
                vehicles_servers_list=None,  # Will be set later
                # tracker = self.tracker,
                env=self.env,
            )

            # Link edge server to cloud server
            edge_server.cloud_server = self.cloud_server
            edge_server.setPPList(edge_platforms)
            edge_server.setModelList(edge_models)
            edge_servers.append(edge_server)
            # Register server with tracker
            self.tracker.register_server(edge_server)

            print(
                f"[SETUP] Created {len(edge_servers)} edge servers with {self.EDGE_COVERAGE_RADIUS}km coverage"
            )
        return edge_servers

    def setup_vehicles(self) -> List[VehicleServer]:
        """Initialize vehicles with routes and processing capabilities."""
        # n print("[SETUP] Creating {self.NUM_VEHICLES} vehicles...")

        vehicles = []

        for i in range(self.NUM_VEHICLES):
            # Select random destination for each vehicle
            destination = self.ROUTE_DESTINATIONS[i]
            vehicle_position = self.VEHICLE_POSITIONS[i]
            # Get route plan from OSRM
            route_plan = VehicleServer.getPath_osrm(vehicle_position, destination)

            if not route_plan:
                # n print("[WARNING] Failed to get route for vehicle {i}, using simple route")
                # Fallback to simple route
                route_plan = [
                    {
                        "coords": (vehicle_position.latitude, vehicle_position.longitude),
                        "distance": 1000,
                        "duration": 72,
                        "road_type": "primary",
                    },
                    {
                        "coords": (destination.latitude, destination.longitude),
                        "distance": 0,
                        "duration": 0,
                        "road_type": "primary",
                    },
                ]

            # Load vehicle processing platforms and models
            vehicle_platforms = load_processing_platforms(
                os.path.join(_DATA_DIR, "processing_platforms.json"), "V", self.env
            )
            vehicle_models = load_dnn_models(os.path.join(_DATA_DIR, "dnn_models.json"), "V")

            # Create edge servers dictionary for vehicle
            edge_servers_dict = {str(edge.id): edge for edge in self.edge_servers}

            # Create vehicle
            vehicle = VehicleServer(
                level=Level.DEVICE,
                brand=f"Vehicle-{i}",
                speed=self.VEHICLE_SPEED,  # km/h
                fps=self.VEHICLE_FPS,
                task_generation_rate=self.VEHICLE_TASK_RATE,
                location=vehicle_position,
                destination_location=destination,
                route_plan=route_plan,
                processing_platforms_list=vehicle_platforms,
                initial_edge_server=str(self.edge_servers[0].id),  # Connect to first edge server
                cloud_server=self.cloud_server,  # Connect to first edge server
                edge_servers_list=edge_servers_dict,
                deployed_models=vehicle_models,
                global_tasks_details=self.global_task_details,
                power_of_vehicle=5000.0,  # Watts
                ram_capacity=8.0,  # GB
                aviable_ram=6.0,  # GB
                bandwidth=100.0,  # Mbps
                network=self.network,
                power_v_e = 20,
                # tracker = self.tracker,
                env=self.env,
            )
            vehicle.setPPList(vehicle_platforms)
            vehicle.setModelList(vehicle_models)
            vehicles.append(vehicle)
            # Register vehicle with tracker
            self.tracker.register_server(vehicle)
        # link to edge servers the list of vehicles
        for edge in self.edge_servers:
            edge.vehicles_servers_list = vehicles
        # n print("[SETUP] Created {len(vehicles)} vehicles with routes in Paris")
        return vehicles

    def setup_simulation(self):
        """Setup all simulation components."""
        print("=" * 60)
        print("EDGE COMPUTING SIMULATION SETUP")
        print("=" * 60)

        # Setup simulation components in order
        self.cloud_server = self.setup_cloud_server()
        self.edge_servers = self.setup_edge_servers()
        self.vehicles = self.setup_vehicles()

        print("\n[SETUP] Simulation components initialized successfully!")
        # n print("[SETUP] - Cloud Server: {self.cloud_server.name}")
        # n print("[SETUP] - Edge Servers: {len(self.edge_servers)}")
        # n print("[SETUP] - Vehicles: {len(self.vehicles)}")
        # n print("[SETUP] - Network: {self.network.__class__.__name__}")

    def run_simulation(self):
        """Execute the main simulation."""
        print("\n" + "=" * 60)
        print("STARTING EDGE COMPUTING SIMULATION")
        print("=" * 60)
        # n print("Duration: {self.SIMULATION_DURATION}s")
        # # n print("Task Generation Rate: {TASK_GENERATION_RATE} tasks/s")
        # n print("Edge Coverage: {self.EDGE_COVERAGE_RADIUS}km")
        print("=" * 60)

        # Track simulation start
        self.tracker.track_simulation_start(self.env.now)

        # Start simulation processes
        start_time = time.time()

        # Add vehicle position update processes
        for vehicle in self.vehicles:
            self.env.process(self.update_vehicle_position(vehicle))

        # Add monitoring process
        self.env.process(self.simulation_monitor())

        # Run simulation
        self.env.run(until=self.SIMULATION_DURATION)
        
        # Track simulation end
        self.tracker.track_simulation_end(self.env.now)

        # Simulation completed
        end_time = time.time()
        execution_time = end_time - start_time

        print("\n" + "=" * 60)
        print("SIMULATION COMPLETED")
        print("=" * 60)
        # n print("Simulation Time: {self.SIMULATION_DURATION}s")
        # n print("Real Execution Time: {execution_time:.2f}s")
        # n print("Time Ratio: {self.SIMULATION_DURATION/execution_time:.2f}x")

        self.print_simulation_results()
        # self.tracker.dump_to_json()
        
        # Export tracking data and print summary
        self.tracker.export_to_files()
        self.tracker.print_summary()

    def update_vehicle_position(self, vehicle: VehicleServer):
        """Continuously update vehicle position during simulation."""
        while True:
            vehicle.update_position()
            # Track vehicle position update
            self.tracker.track_vehicle_position_update(vehicle, self.env.now)

            yield self.env.timeout(0.1)  # Update every 100ms

    def simulation_monitor(self):
        """Monitor simulation progress and print status updates."""
        while True:
            current_time = self.env.now
            progress = (current_time / self.SIMULATION_DURATION) * 100

            # Count active tasks across all vehicles
            total_tasks = sum(len(vehicle.tasks_list) for vehicle in self.vehicles)
            total_current_tasks = sum(
                len(vehicle.current_tasks_data)
                for vehicle in self.vehicles
                if hasattr(vehicle, "current_tasks_data")
            )

            print(
                f"[TIME {current_time:.2f}s] Progress: {progress:.1f}% | Active Tasks: {total_current_tasks} | Total Generated: {total_tasks}"
            )
            # Track server states periodically
            for server in [*self.vehicles, *self.edge_servers]:
                if self.cloud_server:
                    self.tracker.track_server_state_update(self.cloud_server, self.env.now)
                self.tracker.track_server_state_update(server, self.env.now)
            yield self.env.timeout(1.0)  # Update every second

    def print_simulation_results(self):
        """Print comprehensive simulation results."""
        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)

        # Vehicle statistics
        for i, vehicle in enumerate(self.vehicles):
            # n print("\nVehicle {i+1} ({vehicle.name}):")
            # n print("  Final Position: {vehicle.position}")
            # n print("  Trip Completed: {vehicle.trip_finished}")
            # n print("  Total Tasks Generated: {len(vehicle.tasks_list)}")

            # Task statistics
            if hasattr(vehicle, "task_history"):
                completed = len(vehicle.task_history.get("completed", []))
                failed = len(vehicle.task_history.get("failed", []))
                active = len(vehicle.task_history.get("active", []))
                success_rate = completed / (completed + failed) if (completed + failed) > 0 else 0

                # n print("  Tasks Completed: {completed}")
                # n print("  Tasks Failed: {failed}")
                # n print("  Tasks Active: {active}")
                # n print("  Success Rate: {success_rate:.2f}")

        # Edge server statistics
        for i, edge in enumerate(self.edge_servers):
            print("\nEdge Server {i+1} ({edge.name}):")
            # n print("  Location: {edge.location.id} - {edge.location.city}")
            # n print("  Coverage: {edge.length/1000:.1f}km")
            # n print("  Queue Capacity: {edge.task_queue.capacity}")
            # n print("  Current Queue Size: {len(edge.task_queue.items)}")

        # Cloud server statistics
        if self.cloud_server:
            print("\nCloud Server ({self.cloud_server.name}):")
            # n print("  RAM Capacity: {self.cloud_server.ram_capacity}GB")
            # n print("  Available RAM: {self.cloud_server.available_ram}GB")
            # n print("  Processing Platforms: {len(self.cloud_server.processing_platforms_list)}")


# def main():
#     """Main function to run the edge computing simulation."""
#     print("Initializing Edge Computing Simulation...")

#     # Create and setup simulation
#     simulation = EdgeComputingSimulation()
#     simulation.setup_simulation()

#     # Run the simulation
#     simulation.run_simulation()

#     print("\nSimulation finished successfully!")


# if __name__ == "__main__":
#     main()
