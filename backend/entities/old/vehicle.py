# entities/vehicle.py
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simpy import Environment
from entities.task import Task
from entities.processing_platform import ProcessingPlatform, CPU, GPU, NPU, TPU
from entities.dnn_models import DnnModel
from entities.location import Location
import requests
from typing import Any
import folium
import simpy
import json


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
        if processor_type == "cpu":
            processors.append(CPU(**specs))
        elif processor_type == "gpu":
            processors.append(GPU(**specs))
        elif processor_type == "npu":
            processors.append(NPU(**specs))
        elif processor_type == "tpu":
            processors.append(TPU(**specs))
    return processors


class Vehicle:
    idx = 0

    def __init__(
        self,
        brand: str,
        speed: float,
        fps: float,
        tasks_list: list[Task],
        current_location: Location,
        destination_location: Location,
        route_plan: list[dict[str, Any]],
        processing_platforms_list: list,
        deployed_models: list,
        ram_memory: float,
        bandwidth: str,
        env: Environment,
    ):
        Vehicle.idx += 1
        self.id = Vehicle.idx
        self.name = f"Vehicle-{brand}-{self.id}"
        self.speed = speed / 3.6  # Convert km/h to m/s
        self.fps = fps
        self.tasks_list = tasks_list
        self.current_location = current_location
        self.destination_location = destination_location
        self.route_plan = route_plan or []
        self.trip_coordinates: list[tuple] = []
        self.trip_finished = False
        self.processing_platforms_list = processing_platforms_list
        self.deployed_models = deployed_models
        self.ram_memory = ram_memory
        self.bandwidth = float(
            bandwidth.replace("Mbps", "e6").replace("Gbps", "e9")
        )  # Convert to bits/s
        self.env = env
        self.position = (
            [current_location.latitude, current_location.longitude] if current_location else [0, 0]
        )

    def setTaskList(self, tasks_list: list[Task]):
        for task in tasks_list:
            if task not in self.tasks_list:
                task.vehicle = self
                self.tasks_list.append(task)
                print(f"[INFO] Vehicle-setTaskList: Task {task.name} added to Vehicle {self.name}")

    def setPPList(self, processing_platforms_list: list[ProcessingPlatform]):
        for pp in processing_platforms_list:
            if pp not in self.processing_platforms_list:
                pp.current_vehicle = self
                self.processing_platforms_list.append(pp)
                print(
                    f"[INFO] Vehicle-setPPList: Processing Unit {pp.name} added to Vehicle {self.name}"
                )

    def setModelList(self, models: list):
        self.deployed_models = models
        for model in models:
            model.vehicle = self
            print(
                f"[INFO] Vehicle-setModelList: Model {model.name} deployed on Vehicle {self.name}"
            )

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
                print(f"[ERROR] OSRM routing failed: {json_data.get('message')}")
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
            print(f"[ERROR] Failed to fetch route from OSRM: {e}")
            return []

    def update_position(self):
        """Update vehicle position along route_plan (placeholder for Task 6)."""
        if not self.route_plan:
            return
        current_waypoint = self.route_plan[0]
        self.position = list(current_waypoint["coords"])
        print(f"[TIME {self.env.now:.2f}] Vehicle at {self.position}")

    def visualize_route(self, output_file: str = "route_map.html"):
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
        print(f"[INFO] Route map saved to {output_file}")


if __name__ == "__main__":
    env = simpy.Environment()
    vehicle_dnn_models = load_dnn_models("Data/dnn_models.json", "V")
    vehicle_processing_platforms = load_processing_platforms(
        "Data/processing_platforms.json", "V", env
    )
    city = Location("Paris", 48.8566, 2.3522)
    destination_location = Location("Paris-Destination", 48.8666, 2.3622)
    vehicle = Vehicle(
        brand="Tesla",
        speed=120,
        fps=30,
        tasks_list=[],
        current_location=city,
        destination_location=Location("Paris-Destination", 48.8666, 2.3622),
        route_plan=Vehicle.getPath_osrm(city, Location("Paris-Destination", 48.8666, 2.3622)),
        processing_platforms_list=vehicle_processing_platforms,
        deployed_models=vehicle_dnn_models,
        ram_memory=16e9,
        bandwidth="100Mbps",
        env=env,
    )
    route_plan = vehicle.getPath_osrm(city, destination_location)
    vehicle.visualize_route("vehicle_route_map.html")
    print(route_plan)
