from entities.servers.base_server import BaseServer, Level
from entities.servers.edge_server import EdgeServer
from entities.servers.cloud_server import CloudServer
from entities.dnn_models import DnnModel
from entities.location import Location
from entities.processing_platform import ProcessingPlatform, CPU, GPU, NPU, TPU
from entities.cost_model import CostModel
from entities.task_model_platform import TaskModelPlatform, ModelPlatformExecution
from entities.task import Task, TaskStatus
from entities.network import Network
from simpy import Environment, Store
from typing import Any, Optional
import folium
import simpy
import json
import numpy as np
import requests
import random
import math

from entities.simulation_tracker_exteded import get_tracker, SimulationTracker

# VEHICLE_TASK_RATE = 10
TASK_TYPES = [
    "DO",  # Object Detection
    "CI",  # Image Classification
    "S",  # Image Segmentation
    "OT",  # Object tracking
    "TLD",  # Trafic Light Detection
]
TASK_META_DATA_SIZE = 0.16 # a constant estimated prior developement (all tasks generated have almost the same size=16kb)

def load_dnn_models(file_path: str, layer: str):
    with open(file_path, "r") as file:
        models = json.load(file)
    if layer == "V":
        layer_data = models.get("vehicle", [])
    elif layer == "E":
        layer_data = models.get("edge", [])
    elif layer == "C":
        layer_data = models.get("cloud", [])
    else:
        raise ValueError("Layer must be 'V', 'E', or 'C'")
    # n print("[INFO] Loaded {len(layer_data)} models for layer {layer}")
    return [DnnModel(**model) for model in layer_data]


def load_processing_platforms(file_path: str, layer: str, env: simpy.Environment):
    with open(file_path, "r") as file:
        platforms_data = json.load(file)
    if layer == "V":
        layer_data = platforms_data.get("vehicle", {})
    elif layer == "E":
        layer_data = platforms_data.get("edge", {})
    elif layer == "C":
        layer_data = platforms_data.get("cloud", {})
    else:
        raise ValueError("Layer must be 'V', 'E', or 'C'")
    processors: list[Any] = []
    for processor_type, specs in layer_data.items():
        specs["env"] = env
        if processor_type.startswith("cpu"):
            processors.append(CPU(**specs))
        elif processor_type.startswith("gpu"):
            processors.append(GPU(**specs))
        elif processor_type.startswith("npu"):
            processors.append(NPU(**specs))
        elif processor_type.startswith("tpu"):
            processors.append(TPU(**specs))
    return processors


def get_tasks_details() -> dict[str, dict]:
    return {}


class VehicleServer(BaseServer):
    def __init__(
        self,
        level: Level,
        brand: str,
        speed: float,
        fps: int,
        task_generation_rate:int,
        location: Location,
        destination_location: Location,
        route_plan: list[dict[str, Any]],
        processing_platforms_list: list[ProcessingPlatform],
        initial_edge_server: str,  # here we might work with ID instead of object to reduce computations.
        cloud_server: str,  # here we might work with ID instead of object to reduce computations.
        edge_servers_list: dict[str, EdgeServer],
        deployed_models: list[DnnModel],
        global_tasks_details: dict[str, dict],
        power_of_vehicle: float,
        ram_capacity: float,
        aviable_ram: float,
        bandwidth: float,
        network: Network,
        power_v_e: float,
        env: Environment,
    ):
        super().__init__(
            level,
            location,
            processing_platforms_list,
            deployed_models,
            ram_capacity,
            aviable_ram,
            bandwidth,
            env,
        )

        # Vehicle-specific attributes
        self.name = f"Vehicle-{self.id}"
        self.brand = brand
        self.speed = speed / 3.6  # km/h to m/s
        self.fps = fps
        self.task_generation_rate=task_generation_rate
        self.destination_location = destination_location
        self.route_plan = route_plan or []
        self.trip_coordinates: list = []
        self.trip_finished = False
        self.position = [location.latitude, location.longitude] if location else [0, 0]
        self.latency = 0.0  # Local processing
        # Vehicle-server specific attributes
        self.tasks_list: list[Task] = []
        self.tasks_to_be_sent: list[TaskModelPlatform] = (
            []
        )  # For Resulted task informations to be sent to fog // this can be created as a class
        self.setCostModel("vehicle_cost_model", "V")
        self.current_edge_server = initial_edge_server
        self.current_cloud_server =cloud_server
        self.power_of_vehicle = power_of_vehicle
        # Global system attirbutes
        self.edge_servers = edge_servers_list
        self.network: Network = network
        self.global_task_details = global_tasks_details
        self.power_v_e = power_v_e

        # Get tracker instance
        self.tracker:SimulationTracker = get_tracker()

        # Simpy launch the process
        self.env.process(self.task_generation_process())

    @staticmethod
    def getPath_osrm(start_node: Location, end_node: Location) -> list[dict[str, Any]]:
        """
        Fetch route from OSRM using OpenStreetMap data.
        Returns a list of waypoints with coordinates, distances, durations, and road types.
        """
        # OSRM public API endpoint (no API key required)
        base_url = "http://router.project-osrm.org/route/v1/driving/"
        start = f"{start_node.longitude},{start_node.latitude}"
        end = f"{end_node.longitude},{end_node.latitude}"
        url = f"{base_url}{start};{end}?overview=full&geometries=geojson&steps=true"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise exception for bad status codes
            json_data = response.json()

            if json_data.get("code") != "Ok":
                # n print("[ERROR] OSRM routing failed: {json_data.get('message')}")
                return []

            # Extract route details
            route = json_data["routes"][0]
            geometry = route["geometry"]["coordinates"]  # List of [lon, lat]
            legs = route["legs"][0]["steps"]  # Detailed steps with distances and durations

            # Build route_plan with waypoint information
            route_plan = []
            for i, coord in enumerate(geometry):
                lat, lon = coord[1], coord[0]  # Convert [lon, lat] to (lat, lon)
                waypoint = {
                    "coords": (lat, lon),
                    "distance": 0.0,  # Distance to next waypoint (meters)
                    "duration": 0.0,  # Travel time to next waypoint (seconds)
                    "road_type": "unknown",  # Road type (e.g., highway)
                }

                # Assign distance, duration, and road type from steps
                for step in legs:
                    step_coords = step["geometry"]["coordinates"]
                    if [lon, lat] in [[c[0], c[1]] for c in step_coords]:
                        waypoint["distance"] = step.get("distance", 0.0)
                        waypoint["duration"] = step.get("duration", 0.0)
                        waypoint["road_type"] = step.get("maneuver", {}).get("type", "unknown")
                        break

                route_plan.append(waypoint)

            # Normalize distances and durations for consecutive waypoints
            for i in range(len(route_plan) - 1):
                route_plan[i]["distance"] = route_plan[i + 1]["distance"]
                route_plan[i]["duration"] = route_plan[i + 1]["duration"]
            route_plan[-1]["distance"] = 0.0  # Last waypoint has no next distance
            route_plan[-1]["duration"] = 0.0  # Last waypoint has no next duration

            print(
                f"[INFO] Route fetched: {len(route_plan)} waypoints, total distance {route['distance']:.2f}m, duration {route['duration']:.2f}s"
            )
            return route_plan

        except requests.exceptions.RequestException as e:
            # n print("[ERROR] Failed to fetch route from OSRM: {e}")
            return []

    def update_position(self):
        """Update vehicle position along route_plan based on speed and simulation time."""
        if not self.route_plan or self.trip_finished:
            return

        total_distance_traveled = self.speed * self.env.now  # Distance in meters
        accumulated_distance = 0.0
        current_waypoint_index = 0

        # Find the current waypoint based on traveled distance
        for i, waypoint in enumerate(self.route_plan[:-1]):
            next_waypoint = self.route_plan[i + 1]
            segment_distance = waypoint["distance"]
            accumulated_distance += segment_distance
            if accumulated_distance > total_distance_traveled:
                current_waypoint_index = i
                break
            else:
                # Reached the end of the route
                self.position = list(self.route_plan[-1]["coords"])
                self.trip_finished = True
                print(
                    f"TIME {self.env.now:.3f}:  Vehicle {self.name} reached destination at {self.position}"
                )
                return

        # Interpolate position between current and next waypoint
        current_waypoint = self.route_plan[current_waypoint_index]
        next_waypoint = self.route_plan[current_waypoint_index + 1]
        segment_distance = current_waypoint["distance"]
        distance_along_segment = total_distance_traveled - (accumulated_distance - segment_distance)
        fraction = distance_along_segment / segment_distance if segment_distance > 0 else 0

        # Linear interpolation of coordinates
        lat1, lon1 = current_waypoint["coords"]
        lat2, lon2 = next_waypoint["coords"]
        new_lat = lat1 + (lat2 - lat1) * fraction
        new_lon = lon1 + (lon2 - lon1) * fraction
        self.position = [new_lat, new_lon]
        # self.tracker.track_server_action(self, "position_updated", {"new_position": self.position, "trip_finished": self.trip_finished})
        # n print("TIME {self.env.now:.3f}:  Vehicle {self.name} at position {self.position}")
        # NEW: Check for handoff after position update
        self._check_and_perform_handoff()

    # NEW: Handoff detection and execution
    def _check_and_perform_handoff(self):
        """Check if out of current edge range and perform handoff if needed"""
        if not self.current_edge_server or not self.edge_servers:
            # n print("[HANDOFF] Vehicle {self.id} has no current edge or edge list")
            return

        current_edge = self.edge_servers[self.current_edge_server]
        # Calculate distance to current edge (Euclidean, approximate degrees to km)
        dx = self.position[1] - current_edge.location.longitude
        dy = self.position[0] - current_edge.location.latitude
        distance_km = math.sqrt((dx * 111)**2 + (dy * 111 * math.cos(math.radians(self.position[0])))**2)

        if distance_km > (current_edge.length / 1000):  # length in meters, convert to km
            # n print("[HANDOFF DETECT] Vehicle {self.id} out of range from Edge {current_edge.id} (dist: {distance_km:.2f} km)")

            # Find closest edge in range
            closest_edge = None
            min_dist = float('inf')
            old_edge_id = current_edge.id

            for edge in self.edge_servers.values():
                if edge == current_edge:
                    continue
                dx = self.position[1] - edge.location.longitude
                dy = self.position[0] - edge.location.latitude
                dist_km = math.sqrt((dx * 111)**2 + (dy * 111 * math.cos(math.radians(self.position[0])))**2)

                if dist_km <= (edge.length / 1000) and dist_km < min_dist:
                    min_dist = dist_km
                    closest_edge = edge

            if closest_edge:
                # Remove from old edge
                if self in current_edge.vehicles_servers_list:
                    current_edge.vehicles_servers_list.remove(self)
                    # n print("[HANDOFF] Removed Vehicle {self.id} from Edge {old_edge_id}")

                # Add to new edge
                closest_edge.vehicles_servers_list.append(self)
                self.current_edge_server = closest_edge

                # Track handoff
                if self.tracker:
                    self.tracker.track_vehicle_handoff(
                        vehicle_id=self.id,
                        from_edge_id=old_edge_id,
                        to_edge_id=closest_edge.id,
                        timestamp=self.env.now,
                        position=self.position
                    )
                # n print("[HANDOFF] Vehicle {self.id} switched from Edge {old_edge_id} to {closest_edge.id} (dist: {min_dist:.2f} km)")
            else:
                print("[HANDOFF] Vehicle {self.id} out of range, no new edge found (staying with Edge {current_edge.id})")
                # Optional: Fallback to cloud (not implemented)

    def visualize_route(self, output_file: str = "route_map.html"):
        """Visualize the vehicle's route and current position on an interactive map using Folium."""
        if not self.route_plan:
            print("[ERROR] No route plan available to visualize.")
            return

        start_coords = self.route_plan[0]["coords"]
        m = folium.Map(location=start_coords, zoom_start=14, tiles="OpenStreetMap")

        # Plot route as a polyline
        route_coords = [waypoint["coords"] for waypoint in self.route_plan]
        folium.PolyLine(
            locations=route_coords, color="blue", weight=5, opacity=0.7, popup="Vehicle Route"
        ).add_to(m)

        # Add markers for waypoints
        # for i, waypoint in enumerate(self.route_plan):
        #     coords = waypoint["coords"]
        #     distance = waypoint["distance"]
        #     duration = waypoint["duration"]
        #     road_type = waypoint["road_type"]
        #     popup_text = (
        #         f"Waypoint {i}<br>"
        #         f"Lat: {coords[0]:.6f}, Lon: {coords[1]:.6f}<br>"
        #         f"Distance to next: {distance:.2f}m<br>"
        #         f"Duration to next: {duration:.2f}s<br>"
        #         f"Road type: {road_type}"
        #     )
        #     color = "green" if i == 0 else "red" if i == len(self.route_plan) - 1 else "blue"
        #     folium.Marker(
        #         location=coords, popup=popup_text, icon=folium.Icon(color=color, icon="circle")
        #     ).add_to(m)

        # Add marker for current vehicle position
        folium.Marker(
            location=self.position,
            popup=f"Vehicle {self.name}<br>Position: {self.position}<br>Time: {self.env.now:.2f}s",
            icon=folium.Icon(color="purple", icon="car"),
        ).add_to(m)

        m.save(output_file)
        # n print("[INFO] Route map saved to {output_file}")

    def simple_update_position(self):
        """Update vehicle position along route_plan (placeholder for Task 6)."""
        if not self.route_plan:
            return
        current_waypoint = self.route_plan[0]
        self.position = list(current_waypoint["coords"])
        # n print("TIME {self.env.now:.3f}:  Vehicle at {self.position}")

    def simple_visualize_route(self, output_file: str = "route_map.html"):
        """
        Visualize the vehicle's route on an interactive map using Folium.
        Saves the map as an HTML file.
        """
        if not self.route_plan:
            print("[ERROR] No route plan available to visualize.")
            return

        # Initialize map centered on the start location
        start_coords = self.route_plan[0]["coords"]
        m = folium.Map(location=start_coords, zoom_start=14, tiles="OpenStreetMap")

        # Plot route as a polyline
        route_coords = [waypoint["coords"] for waypoint in self.route_plan]
        folium.PolyLine(
            locations=route_coords, color="blue", weight=5, opacity=0.7, popup="Vehicle Route"
        ).add_to(m)

        # Add markers for waypoints
        for i, waypoint in enumerate(self.route_plan):
            coords = waypoint["coords"]
            distance = waypoint["distance"]
            duration = waypoint["duration"]
            road_type = waypoint["road_type"]
            popup_text = (
                f"Waypoint {i}<br>"
                f"Lat: {coords[0]:.6f}, Lon: {coords[1]:.6f}<br>"
                f"Distance to next: {distance:.2f}m<br>"
                f"Duration to next: {duration:.2f}s<br>"
                f"Road type: {road_type}"
            )
            color = "green" if i == 0 else "red" if i == len(self.route_plan) - 1 else "blue"
            folium.Marker(
                location=coords, popup=popup_text, icon=folium.Icon(color=color, icon="circle")
            ).add_to(m)

        # Add marker for current vehicle position
        folium.Marker(
            location=self.position,
            popup=f"Vehicle {self.name}<br>Position: {self.position}",
            icon=folium.Icon(color="purple", icon="car"),
        ).add_to(m)

        # Save map to HTML file
        m.save(output_file)
        # n print("[INFO] Route map saved to {output_file}")

    def setTaskList(self, tasks_list: list[Task]):
        for task in tasks_list:
            if task not in self.tasks_list:
                task.vehicle = self
                self.tasks_list.append(task)
                # n print("[INFO] Vehicle-setTaskList: Task {task.id} added to Vehicle {self.name}")

    def task_generation_process(self):
        """Generate tasks at random intervals using Poisson process."""
        # n print("TIME {self.env.now:.3f}: [INFO] Vehicle {self.id} task generation process started.")
        while True:
            inter_arrival_time = np.random.exponential(1 / self.fps)
            yield self.env.timeout(inter_arrival_time)

            self.generate_vehicle_tasks(TASK_TYPES)

    def generate_vehicle_tasks(self, types_list: list[str]):
        """
        Generate tasks using list of types
        the details of the tasks and the possible types are found in a global variables
        output :
        --- List :  task_model_platform
        """
        # # n print("[INFO] Vehicle {self.id} generating tasks of types: {types_list}")
        generated_tasks_list: list[Optional[TaskModelPlatform]] = []
        for _type in random.sample(types_list,self.task_generation_rate):  # number of type is at max 5 types
            generated_task_dict = self.global_task_details[_type]
            generated_task = Task(**generated_task_dict)
            generated_task.arrival_time = self.env.now
            # Track task generation
            if self.tracker:
                self.tracker.track_task_generation(generated_task, self.id, self.env.now)

            # create a list of Model_platform_executions for generate_task
            generated_task_combinations: Optional[TaskModelPlatform] = None
            for model in self.deployed_models:  # number of models per type is at max 3
                if model.type == _type:
                    for platform in self.processing_platforms_list:  # number of platforms is 2
                        if generated_task_combinations is None:
                            generated_task_combinations = TaskModelPlatform(
                                generated_task,
                                ModelPlatformExecution(
                                    self.level, generated_task.id, model, platform
                                ),
                                self,
                            )
                            continue
                        generated_task_combinations.append_execution(
                            ModelPlatformExecution(self.level, generated_task.id, model, platform)
                        )
            if generated_task_combinations is None:
                continue  # handle this case
            
            generated_tasks_list.append(generated_task_combinations)
            # self.tracker.track_task_event(generated_task_combinations.task, "created", {"type": _type, "vehicle_id": self.id})
        # for t in generated_tasks_list:
        #     if t:
        #         self.tracker.track_task_event(t, "generated")
        print(
            f"TIME {self.env.now:.3f}: [INFO] Vehicle {self.id} generated {len(generated_tasks_list)} tasks."
        )
        # # n print("[INFO] Vehicle {self.id} task details: {generated_tasks_list}")
        self.env.process(self.calculate_task_cost_outputs(generated_tasks_list))

    def calculate_task_cost_outputs(
        self, generated_task_data_list: list[Optional[TaskModelPlatform]]
    ):
        """
        Objectif : calculate cost for all possible executions of all tasks and send them to edge.
        Inputs :
        --- List of combinations in form of TaskModelPlatform objects.
        each combination contains a task and a list of executions in form of objcet(model, platform, and the performance on them)
        Outputs :
        --- For every task, we calculate its cost (cost of all executions) and send it to fog to get offloading decision.
        """
        if not self.cost_model:
            print("[ERROR] Cost model is not set. Cannot calculate task costs.")
            return None
        
        for generated_task_data in generated_task_data_list:
            if not generated_task_data:
                continue
            self.cost_model.estimate(generated_task_data,"V")
            # self.cost_model.estimate(generated_task_data)

            # Track cost estimation
            if self.tracker:
                execution_details = []
                for _, execution in generated_task_data.executions_dict['V'][1].items():
                    execution_details.append({
                        "model_name": execution.model.name if execution.model else "unknown",
                        "platform_name": execution.platform.name if execution.platform else "unknown",
                        "execution_time": execution.execution_time,
                        "memory_consumption": execution.memory_consumption,
                        "energy_consumption": execution.energy_consumption,
                        "platform_usage": execution.platform_usage
                    })
                self.tracker.track_cost_estimation(Level.DEVICE, generated_task_data.task_id, self.env.now, execution_details)

            print(
                f"TIME {self.env.now:.3f}: [INFO] Vehicle {self.id} calculated costs for Task {generated_task_data.task_id}, the estimations are: {[(exec.execution_time, exec.memory_consumption,exec.energy_consumption,exec.platform_usage) for _, exec in generated_task_data.executions_dict['V'][1].items()]}"
            )
            self.tasks_to_be_sent.append(generated_task_data)
            # self.tracker.track_task_event(generated_task_data, "cost_estimated", {"vehicle_id": self.id})
            yield from self.send_tasks_to_edge(generated_task_data)

    def send_tasks_to_edge(self, task_model_platform: TaskModelPlatform):
        if self.current_edge_server in self.edge_servers.keys():
            edge_server = self.edge_servers[self.current_edge_server]
            if isinstance(edge_server, EdgeServer):
                # Track task sending
                if self.tracker:
                    self.tracker.track_task_sending(
                        task_model_platform.task_id, self.id, edge_server.id, 
                        self.env.now, TASK_META_DATA_SIZE
                    )

                yield from self.network.transmit(
                    TASK_META_DATA_SIZE, self, self.current_edge_server
                )
                edge_server.receive_tasks(task_model_platform)
                print(
                    f"TIME {self.env.now:.3f}: [INFO] Vehicle {self.id} sent task {task_model_platform.task_id} to Edge Server {edge_server.id}"
                )
            else:
                print(
                    f"[ERROR] Current edge server {self.current_edge_server} is not an EdgeServer."
                )

    def offload_task_to_destination(self, tasks_decision: list[TaskModelPlatform]):
        processes = []
        for task in tasks_decision:
            if task is None or task.chosen_execution is None:
                # n print("[ERROR] Invalid task or chosen_execution: {task}")
                continue
            if task.chosen_execution.level == Level.DEVICE:
                processes.append(
                    self.env.process(self.execute_task_locally(task.task, task.chosen_execution))
                )
            elif task.chosen_execution.level == Level.EDGE:
                processes.append(
                    self.env.process(self.execute_on_edge_server(task.task, task.chosen_execution))
                )
            else:
                processes.append(
                    self.env.process(self.execute_on_cloud_server(task.task, task.chosen_execution))
                )
        if processes:
            yield self.env.all_of(processes)  # Parallel execution
        # n print("TIME {self.env.now:.3f}: [INFO] Offloaded {len(processes)} tasks from Vehicle {self.id}")

    def execute_task_locally(self, task: Task, task_execution: ModelPlatformExecution):
        if task is None or task_execution is None:
            # n print("[ERROR] Task execution failed: Invalid inputs")
            if task:
                task.status = TaskStatus.FAILED
            return
                # Track task execution start
        if self.tracker:
            self.tracker.track_task_execution_start(task.id, self.id, self.env.now, task_execution)
        execution_start_time = self.env.now
        yield self.env.process(self.start_task_execution(task, task_execution))
        execution_end_time = self.env.now
        # Track task execution end
        if self.tracker:
            self.tracker.track_task_execution_end(
                task.id, self.id, execution_end_time, task.status, 
                execution_end_time - execution_start_time
            )

    def execute_on_edge_server(self, task: Task, task_execution: ModelPlatformExecution):
        if task is None or task_execution.model is None:
            # n print("[ERROR] Task {task.id if task else 'None'} execution failed: Invalid inputs")
            if task:
                task.status = TaskStatus.FAILED
            return
        
        if self.current_edge_server in self.edge_servers.keys():
            edge_server = self.edge_servers[self.current_edge_server]
            if isinstance(edge_server, EdgeServer):
                size_of_data_to_be_send = task.data_size + TASK_META_DATA_SIZE
                yield self.env.process(
                    self.network.transmit(
                        size_of_data_to_be_send, str(self.id), self.current_edge_server
                    )
                )
                # Track task execution start
                if self.tracker:
                    self.tracker.track_task_execution_start(task.id, edge_server.id, self.env.now, task_execution)
                execution_start_time = self.env.now
                yield self.env.process(edge_server.start_task_execution(task, task_execution))
                execution_end_time = self.env.now
                # Track task execution end
                if self.tracker:
                    self.tracker.track_task_execution_end(
                        task.id, edge_server.id, execution_end_time, task.status,
                        execution_end_time - execution_start_time
                    )

                print(
                    f"TIME {self.env.now:.3f}: [INFO] Task {task_execution.task_id} started execution in Edge Server {edge_server.id}"
                )
            else:
                print(
                    f"[ERROR] Current edge server {self.current_edge_server} is not an EdgeServer."
                )
                task.status = TaskStatus.FAILED

    def execute_on_cloud_server(self, task: Task, task_execution: ModelPlatformExecution):
        if task is None or task_execution.model is None:
            # n print("[ERROR] Task {task.id if task else 'None'} execution failed: Invalid inputs")
            if task:
                task.status = TaskStatus.FAILED
            return
        
        if self.current_cloud_server and isinstance(self.current_cloud_server, CloudServer):
            size_of_data_to_be_send = task.data_size + TASK_META_DATA_SIZE
            yield self.env.process(
                self.network.transmit(
                    size_of_data_to_be_send, str(self.id), self.current_cloud_server
                )
            )
            # Track task execution start
            if self.tracker:
                self.tracker.track_task_execution_start(task.id, self.current_cloud_server.id, self.env.now, task_execution)
            execution_start_time = self.env.now
            yield self.env.process(self.current_cloud_server.start_task_execution(task, task_execution))
            execution_end_time = self.env.now
            # Track task execution end
            if self.tracker:
                self.tracker.track_task_execution_end(
                    task.id, self.current_cloud_server.id, execution_end_time, task.status,
                    execution_end_time - execution_start_time
                )

            print(
                f"TIME {self.env.now:.3f}: [INFO] Task {task_execution.task_id} started execution in Cloud Server {self.current_cloud_server.id}"
            )
        else:
            print(
                f"[ERROR] Current edge server {self.current_edge_server} is not an EdgeServer."
            )
            task.status = TaskStatus.FAILED

