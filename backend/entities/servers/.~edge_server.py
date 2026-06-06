# entities/cloud_layer.py
from entities.servers.base_server import BaseServer, Level
from entities.location import Location
from entities.processing_platform import ProcessingPlatform, CPU, GPU
from entities.cost_model import CostModel
from entities.task_model_platform import TaskModelPlatform, ModelPlatformExecution
from entities.network import Network
from simpy import Environment, Store, Interrupt
from typing import Any, Optional, TYPE_CHECKING
import queue
import numpy as np
from datetime import datetime, timedelta
from entities.servers.offloading_algorithme import OffloaingAlgorithme, PPOModel
from entities.servers.config import norm_params
import time

if TYPE_CHECKING:
    from entities.servers.cloud_server import CloudServer

TIME_SLOT = 0.1
MAX_BATCH_SIZE = 10
N_MAX = 150
V_MAX = 30
TASK_TYPES = ["DO", "CI", "S", "OT", "TLD"]
MAX_COMBINATIONS_PER_EDGE = 6
HARDWARE_VEHICLE = ["NANO", "INTEL_I7"]
HARDWARE_EDGE = ["TX2", "NANO", "INTEL_I7"]
HARDWARE_CLOUD = ["AGX", "NANO", "TX2", "INTEL_I7"]
HARDWARE_NAMES = ["NANO", "INTEL_I7", "TX2", "AGX"]
MAX_COMBINATIONS_PER_TYPE = 20
MODEL_NAMES = [
    "SSD-MobileNetV3",
    "YOLOv5s",
    "SSD-EfficientNet",
    "YOLOv8x",
    "Mask R-CNN",
    "DETR",
    "MobileNetV3-Small",
    "EfficientNet-B0",
    "ResNet-18",
    "ResNet-50",
    "EfficientNet-B4",
    "ViT-Base",
    "MobileUnet",
    "DeepLabV3-MobileNet",
    "U-Net-Lite",
    "DeepLabV3-ResNet",
    "SegFormer",
    "TinySORT",
    "DeepSORT",
    "FairMOT-Lite",
    "FairMOT",
    "ByteTrack",
    "OC-SORT",
    "YOLOv3-Tiny-TLD",
    "YOLOv5s-TLD",
    "SSD-MobileNet-TLD",
    "YOLOv8m-TLD",
    "Faster R-CNN-TLD",
    "DETR-TLD",
]
LEVELS = ["V", "E", "C"]


class EdgeServer(BaseServer):
    cloud_server: Optional["CloudServer"] = None  # Reference to the cloud server
    ppo_model = PPOModel()

    def __init__(
        self,
        level: Level,
        location: Location,
        length: float,
        processing_platforms_list: list[ProcessingPlatform],
        deployed_models: list,
        ram_capacity: float,
        avilable_ram: float,
        bandwidth: float,
        power_P_e_e: float,
        power_P_e_v: float,
        power_P_e_c: float,
        network:Network,
        env: Environment,
    ):
        super().__init__(
            level,
            location,
            processing_platforms_list,
            deployed_models,
            ram_capacity,
            avilable_ram,
            bandwidth,
            env,
        )
        # EdgeServer specific attributes
        self.name = f"Edge-{self.id}"
        self.latency = 0.01  # Low latency (seconds)
        self.length = length
        self.task_queue = Store(env, MAX_BATCH_SIZE)  # For Task 2: Task queue
        self.setCostModel("edge_cost_model", "E")
        self.power_P_e_e = power_P_e_e
        self.power_P_e_v = power_P_e_v
        self.power_P_e_c = power_P_e_c
        self.network= network
        self.env.process(self.batch_extractor_agent())
    def reveive_tasks(self, task_model_platform: TaskModelPlatform):
        self.add_task_to_queue(task_model_platform)
        print(f"[INFO] Task {task_model_platform.task_id} received by Edge {self.name}")

    def add_task_to_queue(self, task: TaskModelPlatform):
        task.position_in_edge_queue = int(self.task_queue.capacity - len(self.task_queue.items))
        task.arrival_time_to_edge_queue = self.env.now
        self.task_queue.put(task)
        print(f"[INFO] Task {task.task_id} added to {self.name} queue")

    def _extract_tasks_from_store(self, max_tasks: int):
        """
        Synchronously extracts up to max_tasks from the store.
        This is not a process itself, but a helper function called by one.
        """
        tasks_extracted: list[Optional[TaskModelPlatform]] = []

        # Extract tasks as long as the store is not empty and the batch isn't full
        for _ in range(max_tasks):
            if not self.task_queue.items:
                # Stop if the queue is empty
                break
            # We use yield here because get() is a generator-based operation in SimPy
            task = yield self.task_queue.get()
            task.extraction_time_to_edge_queue = self.env.now
            # --- You can set task properties here ---
            # For example: task.extraction_time = self.env.now
            # For example: task.status = "Ready for offloading"

            tasks_extracted.append(task)

        return tasks_extracted

    def batch_extractor_agent(self):
        """
        A continuous SimPy process that wakes up every TIME_SLOT,
        and extracts a batch of tasks from the queue.
        """
        print(f"Time {self.env.now:.2f}: [AGENT] Batch extractor agent started for {self.name}.")
        while True:
            try:
                # 1. Wait for the defined time slot before attempting extraction.
                print(f"\nTime {self.env.now:.2f}: [AGENT] Agent is sleeping for {TIME_SLOT}ms.")
                yield self.env.timeout(TIME_SLOT)

                # 2. After waking up, check if there are tasks to process.
                if not self.task_queue.items:
                    print(
                        f"Time {self.env.now:.2f}: [AGENT] Agent woke up. No tasks in queue. Going back to sleep."
                    )
                    continue

                print(
                    f"Time {self.env.now:.2f}: [AGENT] Agent woke up. Attempting to extract batch of up to {MAX_BATCH_SIZE} tasks."
                )

                # 3. Extract the batch of tasks.
                # We use 'yield from' if the called function is a generator, or env.process if it's a full process.
                # For simplicity, let's make the helper a standard generator.
                batch_process = self.env.process(self._extract_tasks_from_store(MAX_BATCH_SIZE))
                batch = yield batch_process

                # 4. Process the extracted batch
                if batch:
                    task_ids = [task.task_id for task in batch]
                    print(
                        f"Time {self.env.now:.2f}: [AGENT] Successfully extracted batch of {len(batch)} tasks (IDs: {task_ids})."
                    )
                    print(
                        f"Time {self.env.now:.2f}: [AGENT] Remaining tasks in queue: {len(self.task_queue.items)}"
                    )
                    # --- MARL Processing would happen here ---
                    self.env.process(self.generate_edge_tasks(batch))
                else:
                    print(
                        f"Time {self.env.now:.2f}: [AGENT] Woke up, but no tasks were extracted in this cycle."
                    )

            except Interrupt as i:
                print(
                    f"Time {self.env.now:.2f}: [AGENT] Batch extractor agent interrupted: {i.cause}"
                )
                break

    def generate_edge_tasks(self, tasks_batch: list[Optional[TaskModelPlatform]]):
        """
        Generate tasks using list of types
        the details of the tasks and the possible types are found in a global variables
        output :
        --- List :  task_model_platform
        """
        for task in tasks_batch:  # number of type is at max 5 types
            if not task:
                continue
            _type = task.task.type
            task_id = task.task.id
            # append execution of models and platforms of the EdgeServer in task
            for model in self.deployed_models:  # number of models per type is at max 3
                if model.type == _type:
                    for platform in self.processing_platforms_list:  # number of platforms is 2
                        task.append_execution(
                            ModelPlatformExecution(self.level, task_id, model, platform)
                        )
            # append execution of models and platforms of the cloudServer in task
            if self.cloud_server:
                for model in self.cloud_server.deployed_models:
                    if model.type == _type:
                        for platform in self.cloud_server.processing_platforms_list:
                            task.append_execution(
                                ModelPlatformExecution(
                                    self.cloud_server.level, task_id, model, platform
                                )
                            )

            # Calculate future time and position for alpha
            self.calculate_future_time(task, self.network)  # type: ignore
            self.calculate_future_position(task)
        yield from self.calculate_task_cost_outputes(tasks_batch)

    def calculate_task_cost_outputes(
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

        print(
            f"[TIME {self.env.now:.2f}] Starting cost calculation and offloading decisions for {len(generated_task_data_list)} tasks"
        )

        for tasks_batch in generated_task_data_list:
            if not tasks_batch:
                continue
            self.cost_model.estimate(tasks_batch)
            if self.cloud_server and self.cloud_server.cost_model:
                self.cloud_server.cost_model.estimate(tasks_batch)

        print(f"[TIME {self.env.now:.2f}] Cost estimation completed for all tasks")

        # 2. Prepare inputs for offloading algorithm
        start_time = time.time()
        rl_inputs = self.prepare_offloading_inputs(generated_task_data_list)
        input_preparation_time = time.time() - start_time

        print(f"[TIME {self.env.now:.2f}] RL inputs prepared in {input_preparation_time:.4f}s")

        # Simulate the time needed for input preparation
        yield self.env.timeout(input_preparation_time)

        # 3. Calculate offloading decisions using RL algorithm
        print(f"[TIME {self.env.now:.2f}] Starting offloading decision calculation...")

        decision_start_time = time.time()
        tasks_with_decisions = yield from self._calculate_offloading_decisions_process(
            generated_task_data_list, rl_inputs
        )
        decision_time = time.time() - decision_start_time

        print(f"[TIME {self.env.now:.2f}] Offloading decisions completed in {decision_time:.4f}s")

        # 4. Send results back to vehicles
        yield from self._send_decisions_to_vehicles(tasks_with_decisions)

        print(f"[TIME {self.env.now:.2f}] All tasks processed and sent back to vehicles")

    def calculate_future_time(self, task_model_platform: TaskModelPlatform, network: Network):
        """
        Calculate the estimated time by which the task execution would be complete
        """
        for _, level_execution in task_model_platform.executions_dict.items():
            for __, execution in level_execution[1].items():
                time_after_execution = (
                    (
                        task_model_platform.extraction_time_to_edge_queue
                        - task_model_platform.arrival_time_to_edge_queue
                    )
                    + ((task_model_platform.position_in_edge_queue or 0) / MAX_BATCH_SIZE)* TIME_SLOT
                    + execution.execution_time
                    # should add time of response transmission
                )
                execution.duration_for_future_position = time_after_execution

    def calculate_future_position(
        self,
        task_model_platform: TaskModelPlatform,
    ):
        """Calculate the vehicle's position at a future time. and calculate Alpha value depending if vehicle exited the edge length"""
        """
        Note:
        --- this should be calculated for every execution and alpha should be updated accordingly
        """

        if not task_model_platform.vehicle.route_plan or task_model_platform.vehicle.trip_finished:
            return

        for _, level_execution in task_model_platform.executions_dict.items():
            for __, execution in level_execution[1].items():
                total_distance_traveled = (
                    task_model_platform.vehicle.speed * execution.duration_for_future_position
                )  # Distance in meters
                accumulated_distance = 0.0
                current_waypoint_index = 0

                # Find the current waypoint based on traveled distance
                for i, waypoint in enumerate(task_model_platform.vehicle.route_plan[:-1]):
                    next_waypoint = task_model_platform.vehicle.route_plan[i + 1]
                    segment_distance = waypoint["distance"]
                    accumulated_distance += segment_distance
                    if accumulated_distance > total_distance_traveled:
                        current_waypoint_index = i
                        break
                else:
                    # Reached the end of the route
                    task_model_platform.vehicle.position = list(
                        task_model_platform.vehicle.route_plan[-1]["coords"]
                    )
                    task_model_platform.vehicle.trip_finished = True
                    print(
                        f"[TIME {self.env.now:.2f}] Vehicle {task_model_platform.vehicle.name} reached destination at {task_model_platform.vehicle.position}"
                    )
                    return

                # Interpolate position between current and next waypoint
                current_waypoint = task_model_platform.vehicle.route_plan[current_waypoint_index]
                next_waypoint = task_model_platform.vehicle.route_plan[current_waypoint_index + 1]
                segment_distance = current_waypoint["distance"]
                distance_along_segment = total_distance_traveled - (
                    accumulated_distance - segment_distance
                )
                fraction = distance_along_segment / segment_distance if segment_distance > 0 else 0

                # Calculate Alpha and compare it with esge Length
                if accumulated_distance > self.length:
                    execution.alpha = 1.0
                else:
                    execution.alpha = 0.0

                # Linear interpolation of coordinates
                lat1, lon1 = current_waypoint["coords"]
                lat2, lon2 = next_waypoint["coords"]
                new_lat = lat1 + (lat2 - lat1) * fraction
                new_lon = lon1 + (lon2 - lon1) * fraction
                execution.future_position = new_lat, new_lon
                print(
                    f"[TIME {self.env.now:.2f}] Vehicle {task_model_platform.vehicle.name} after {execution.duration_for_future_position} will be at position {task_model_platform.vehicle.position}"
                )

    def normalize_value(self, value: float, param_key: str) -> float:
        min_val = norm_params[param_key]["min"]
        max_val = norm_params[param_key]["max"]
        if max_val - min_val == 0:
            return np.array([0.0], dtype=np.float32)  # type: ignore
        normalized_value = (value - min_val) / (max_val - min_val)
        return np.array([np.clip(normalized_value, 0.0, 1.0)], dtype=np.float32)  # type: ignore

    def prepare_offloading_inputs(self, task_batch: list[Optional[TaskModelPlatform]]) -> dict:
        """
        Prepares inputs for the offloading RL system based on the PDF format.
        Args:
            task_batch: List of TaskModelPlatform objects containing tasks and their execution combinations.
        Returns:
            Dict: Input data structured as per the PDF, compatible with Stable-Baselines3 PPO or Gym.
        """
        # Initialize the input dictionary
        inputs = {
            "Tasks_Data": {
                "id_Tache": np.full((N_MAX, 1), -1, dtype=np.int64),
                "Type": np.full((N_MAX, 1), len(TASK_TYPES), dtype=np.int32),
                "min_accuracy": np.full((N_MAX, 1), -1.0, dtype=np.float32),
                "time_to_deadline": np.full((N_MAX, 1), -1.0, dtype=np.float32),
                "data_size_input": np.full((N_MAX, 1), -1.0, dtype=np.float32),
                "data_size_output": np.full((N_MAX, 1), -1.0, dtype=np.float32),
                "alpha_list_edge_comb": np.zeros((N_MAX, MAX_COMBINATIONS_PER_EDGE), dtype=np.int8),
                "vehicle_idx": np.full((N_MAX, 1), -1, dtype=np.int64),
            },
            "Vehicles_Data": {
                "id_Vehicule": np.full((V_MAX, 1), -1, dtype=np.int64),
                "power_v_e": np.full((V_MAX, 1), -1.0, dtype=np.float32),
                "hardware_name_id": np.full(
                    (V_MAX, len(HARDWARE_VEHICLE), 1), len(HARDWARE_NAMES), dtype=np.int64
                ),
                "charge_remaining_percentage": np.full(
                    (V_MAX, len(HARDWARE_VEHICLE), 1), -1.0, dtype=np.float32
                ),
                "memory_capacity_remaining_of_this_hardware": np.full(
                    (V_MAX, len(HARDWARE_VEHICLE), 1), -1.0, dtype=np.float32
                ),
            },
            "Combinations_Data": {},
            "Info_Edge": {
                "power_P_e_e": np.array([-1.0], dtype=np.float32),
                "power_P_e_v": np.array([-1.0], dtype=np.float32),
                "power_P_e_c": np.array([-1.0], dtype=np.float32),
                "hardware_name_id": np.full(
                    (len(HARDWARE_EDGE), 1), len(HARDWARE_NAMES), dtype=np.int64
                ),
                "charge_remaining_percentage": np.full(
                    (len(HARDWARE_EDGE), 1), -1.0, dtype=np.float32
                ),
                "memory_capacity_remaining_of_this_hardware": np.full(
                    (len(HARDWARE_EDGE), 1), -1.0, dtype=np.float32
                ),
            },
            "Info_Cloud": {
                "power_P_c_e": np.array([-1.0], dtype=np.float32),
                "hardware_name_id": np.full(
                    (len(HARDWARE_CLOUD), 1), len(HARDWARE_NAMES), dtype=np.int64
                ),
                "charge_remaining_percentage": np.full(
                    (len(HARDWARE_CLOUD), 1), -1.0, dtype=np.float32
                ),
                "memory_capacity_remaining_of_this_hardware": np.full(
                    (len(HARDWARE_CLOUD), 1), -1.0, dtype=np.float32
                ),
            },
            "Info_Communication": {
                "vehicle_to_edge_throughput": np.array([-1.0], dtype=np.float32),
                "edge_to_edge_throughput": np.array([-1.0], dtype=np.float32),
                "edge_to_cloud_throughput": np.array([-1.0], dtype=np.float32),
                "edge_to_vehicle_throughput": np.array([-1.0], dtype=np.float32),
                "cloud_to_edge_throughput": np.array([-1.0], dtype=np.float32),
            },
        }

        # Track unique vehicles to avoid duplicates
        vehicle_ids: set = set()

        for task_type in TASK_TYPES:
            inputs["Combinations_Data"][task_type] = {  # type: ignore
                "model_name_id": np.full((MAX_COMBINATIONS_PER_TYPE, 1), -1, dtype=np.int32),
                "hardware_name_id": np.full((MAX_COMBINATIONS_PER_TYPE, 1), -1, dtype=np.int32),
                "accuracy": np.full((MAX_COMBINATIONS_PER_TYPE, 1), -1.0, dtype=np.float32),
                "execution_time": np.full((MAX_COMBINATIONS_PER_TYPE, 1), -1.0, dtype=np.float32),
                "power_consumption": np.full(
                    (MAX_COMBINATIONS_PER_TYPE, 1), -1.0, dtype=np.float32
                ),
                "memory_consumption": np.full(
                    (MAX_COMBINATIONS_PER_TYPE, 1), -1.0, dtype=np.float32
                ),
                "utilization_percentage": np.full(
                    (MAX_COMBINATIONS_PER_TYPE, 1), -1.0, dtype=np.float32
                ),
                "niveau_id": np.full((MAX_COMBINATIONS_PER_TYPE, 1), -1, dtype=np.int32),
            }

        for i, task_model_platform in enumerate(task_batch[:N_MAX]):  # Limit to N_max
            if not task_model_platform or not task_model_platform.task:
                continue

            task = task_model_platform.task
            vehicle = task_model_platform.vehicle

            # Fill task data with proper normalization
            inputs["Tasks_Data"]["id_Tache"][i, 0] = task.id  # type: ignore
            inputs["Tasks_Data"]["Type"][i, 0] = (  # type: ignore
                TASK_TYPES.index(task.type) if task.type in TASK_TYPES else len(TASK_TYPES)
            )
            inputs["Tasks_Data"]["min_accuracy"][i, 0] = self.normalize_value(  # type: ignore
                task.min_accuracy, "min_accuracy"
            )
            inputs["Tasks_Data"]["time_to_deadline"][i, 0] = self.normalize_value(  # type: ignore
                task.max_latency, "time_to_deadline"
            )
            inputs["Tasks_Data"]["data_size_input"][i, 0] = self.normalize_value(  # type: ignore
                task.data_size, "data_size_input"
            )
            inputs["Tasks_Data"]["data_size_output"][i, 0] = self.normalize_value(  # type: ignore
                task.data_size_output, "data_size_output"
            )
            inputs["Tasks_Data"]["vehicle_idx"][i, 0] = vehicle.id if vehicle else -1  # type: ignore

            # Extract alpha values for edge combinations
            edge_executions = task_model_platform.executions_dict.get(
                "E", ({"is_calculated": False}, {})
            )
            if edge_executions[0].get("is_calculated", False):
                alpha_list = [execution.alpha for execution in edge_executions[1].values()]
                max_edge_combs = inputs["Tasks_Data"]["alpha_list_edge_comb"].shape[1]  # type: ignore
                for j, alpha in enumerate(alpha_list[:max_edge_combs]):
                    inputs["Tasks_Data"]["alpha_list_edge_comb"][i, j] = int(alpha)  # type: ignore

        # 2. Process Vehicles_Data
        vehicle_list = [
            tmb.vehicle
            for tmb in task_batch
            if tmb and tmb.vehicle and tmb.vehicle.id in vehicle_ids
        ]
        unique_vehicles = {v.id: v for v in vehicle_list}.values()

        for i, vehicle in enumerate(list(unique_vehicles)[:V_MAX]):
            inputs["Vehicles_Data"]["id_Vehicule"][i, 0] = vehicle.id  # type: ignore
            inputs["Vehicles_Data"]["power_v_e"][i, 0] = self.normalize_value(vehicle.power_v_e, "power_P_v_e")  # type: ignore

            # Handle multiple hardware per vehicle
            for j, platform in enumerate(
                vehicle.processing_platforms_list[: len(HARDWARE_VEHICLE)]
            ):
                # Get hardware name ID (index in hardware_names list)
                hw_id = (
                    HARDWARE_NAMES.index(platform.name)
                    if platform.name in HARDWARE_NAMES
                    else len(HARDWARE_NAMES)
                )
                inputs["Vehicles_Data"]["hardware_name_id"][i, j, 0] = hw_id  # type: ignore

                # Calculate charge remaining percentage
                charge_remaining = (
                    (platform.power_efficiency.capacity - platform.power_efficiency.level)
                    / platform.power_efficiency.capacity
                    * 100
                    if platform.power_efficiency.capacity > 0
                    else 0
                )
                inputs["Vehicles_Data"]["charge_remaining_percentage"][i, j, 0] = (  # type: ignore
                    self.normalize_value(charge_remaining, "charge_remaining_percentage")
                )

                if platform.current_server:
                    # Memory capacity remaining
                    inputs["Vehicles_Data"]["memory_capacity_remaining_of_this_hardware"][i, j, 0] = (  # type: ignore
                        self.normalize_value(
                            platform.current_server.available_ram,
                            "memory_capacity_remaining_of_this_hardware",
                        )
                    )

                else:
                    # Memory capacity remaining
                    inputs["Vehicles_Data"]["memory_capacity_remaining_of_this_hardware"][i, j, 0] = (  # type: ignore
                        self.normalize_value(
                            platform.memory_size.level, "memory_capacity_remaining_of_this_hardware"
                        )
                    )

        # 3. Process Combinations_Data
        for task_model_platform in task_batch:
            if not task_model_platform:
                continue

            task_type = task_model_platform.task.type
            if task_type not in TASK_TYPES:
                continue

            # Process all execution combinations for this task type
            comb_idx = 0
            for level, (meta, executions) in task_model_platform.executions_dict.items():
                if not meta.get("is_calculated", False):
                    continue

                for _, execution in executions.items():
                    if comb_idx >= MAX_COMBINATIONS_PER_TYPE:
                        break

                    # Get model and hardware IDs
                    model_id = (
                        MODEL_NAMES.index(execution.model.name)
                        if execution.model.name in MODEL_NAMES
                        else -1
                    )
                    hw_id = (
                        MODEL_NAMES.index(execution.platform.name)
                        if execution.platform.name in HARDWARE_NAMES
                        else -1
                    )
                    niveau_id = LEVELS.index(level) if level in ["V", "E", "C"] else -1

                    # Fill combination data with normalization
                    inputs["Combinations_Data"][task_type]["model_name_id"][comb_idx, 0] = model_id  # type: ignore
                    inputs["Combinations_Data"][task_type]["hardware_name_id"][comb_idx, 0] = hw_id  # type: ignore
                    inputs["Combinations_Data"][task_type]["accuracy"][comb_idx, 0] = (  # type: ignore
                        self.normalize_value(execution.model_accuracy, "accuracy_of_model")
                    )
                    inputs["Combinations_Data"][task_type]["execution_time"][comb_idx, 0] = (  # type: ignore
                        self.normalize_value(execution.execution_time, "execution_time")
                    )
                    inputs["Combinations_Data"][task_type]["power_consumption"][comb_idx, 0] = (  # type: ignore
                        self.normalize_value(execution.energy_consumption, "power_consumption")
                    )
                    inputs["Combinations_Data"][task_type]["memory_consumption"][comb_idx, 0] = (  # type: ignore
                        self.normalize_value(execution.memory_consumption, "memory_consumption")
                    )
                    inputs["Combinations_Data"][task_type]["utilization_percentage"][  # type: ignore
                        comb_idx, 0
                    ] = self.normalize_value(
                        execution.platform_usage, "utilization_percentage"
                    )  # type: ignore
                    inputs["Combinations_Data"][task_type]["niveau_id"][comb_idx, 0] = niveau_id  # type: ignore

                    comb_idx += 1

        # 4. Process Info_Edge
        # Set power values (these may need to be retrieved from edge server properties)
        inputs["Info_Edge"]["power_P_e_e"] = self.normalize_value(  # type: ignore
            self.power_P_e_e, "power_P_e_e"
        )
        inputs["Info_Edge"]["power_P_e_v"] = self.normalize_value(  # type: ignore
            self.power_P_e_v, "power_P_e_v"
        )
        inputs["Info_Edge"]["power_P_e_c"] = self.normalize_value(  # type: ignore
            self.power_P_e_c, "power_P_e_c"
        )

        # Process edge hardware
        for i, platform in enumerate(self.processing_platforms_list[: len(HARDWARE_EDGE)]):
            hw_id = (
                HARDWARE_NAMES.index(platform.name)
                if platform.name in HARDWARE_NAMES
                else len(HARDWARE_NAMES)
            )
            inputs["Info_Edge"]["hardware_name_id"][i, 0] = hw_id  # type: ignore

            charge_remaining = platform.platform_usage
            inputs["Info_Edge"]["charge_remaining_percentage"][i, 0] = self.normalize_value(  # type: ignore
                charge_remaining, "charge_remaining_percentage"
            )
            if platform.current_server:
                inputs["Info_Edge"]["memory_capacity_remaining_of_this_hardware"][i, 0] = (  # type: ignore
                    self.normalize_value(  # type: ignore
                        platform.current_server.available_ram,
                        "memory_capacity_remaining_of_this_hardware",
                    )
                )
            else:
                inputs["Info_Edge"]["memory_capacity_remaining_of_this_hardware"][i, 0] = (  # type: ignore
                    self.normalize_value(  # type: ignore
                        platform.memory_size.level, "memory_capacity_remaining_of_this_hardware"
                    )
                )
        # 5. Process Info_Cloud
        if self.cloud_server:
            inputs["Info_Cloud"]["power_P_c_e"][0] = self.normalize_value(  # type:ignore
                self.cloud_server.power_P_c_e, "power_P_c_e"
            )

            # Process cloud hardware
            for i, platform in enumerate(
                self.cloud_server.processing_platforms_list[: len(HARDWARE_CLOUD)]
            ):
                hw_id = (
                    HARDWARE_NAMES.index(platform.name)
                    if platform.name in HARDWARE_NAMES
                    else len(HARDWARE_NAMES)
                )
                inputs["Info_Cloud"]["hardware_name_id"][i, 0] = hw_id  # type:ignore

                charge_remaining = (
                    (platform.power_efficiency.capacity - platform.power_efficiency.level)
                    / platform.power_efficiency.capacity
                    * 100
                    if platform.power_efficiency.capacity > 0
                    else 0
                )
                inputs["Info_Cloud"]["charge_remaining_percentage"][i, 0] = (  # type:ignore
                    self.normalize_value(charge_remaining, "charge_remaining_percentage")
                )
                inputs["Info_Cloud"]["memory_capacity_remaining_of_this_hardware"][i, 0] = (
                    self.normalize_value(
                        platform.memory_size.level, "memory_capacity_remaining_of_this_hardware"
                    )
                )

        # 6. Process Info_Communication
        if self.network:
            inputs["Info_Communication"]["vehicle_to_edge_throughput"][0] = (  # type:ignore
                self.normalize_value(
                    self.network.vehicle_to_edge_throughput, "vehicle_to_edge_throughput"
                )
            )
            inputs["Info_Communication"]["edge_to_edge_throughput"] = (  # type:ignore
                self.normalize_value(
                    self.network.edge_to_edge_throughput, "edge_to_edge_throughput"
                )
            )
            inputs["Info_Communication"]["edge_to_cloud_throughput"] = (  # type:ignore
                self.normalize_value(
                    self.network.edge_to_cloud_throughput, "edge_to_cloud_throughput"
                )
            )
            inputs["Info_Communication"]["edge_to_vehicle_throughput"] = (  # type:ignore
                self.normalize_value(
                    self.network.edge_to_vehicle_throughput, "edge_to_vehicle_throughput"
                )
            )
            inputs["Info_Communication"]["cloud_to_edge_throughput"] = (  # type:ignore
                self.normalize_value(
                    self.network.cloud_to_edge_throughput, "cloud_to_edge_throughput"
                )
            )

        return inputs

    def _calculate_offloading_decisions_process(
        self, task_batch: list[Optional[TaskModelPlatform]], rl_inputs, algorithme: str = "RL"
    ):
        """
        SimPy process to calculate offloading decisions using the RL algorithm
        """
        # Initialize the offloading algorithm (PPO model)
        if algorithme == "RL":
            try:
                print(
                    f"[TIME {self.env.now:.2f}] Initializing PPO model for offloading decisions..."
                )
                if not self.ppo_model:
                    self.ppo_model = PPOModel()

                algorithm_start_time = time.time()
                # Calculate decisions
                tasks_with_decisions = self.ppo_model.calculate_offloading_decisions(
                    task_batch, rl_inputs
                )

                algorithm_execution_time = time.time() - algorithm_start_time
                print(
                    f"[TIME {self.env.now:.2f}] PPO algorithm executed in {algorithm_execution_time:.4f}s"
                )

                # Simulate the computational time in the simulation
                yield self.env.timeout(algorithm_execution_time)

                # Validate decisions
                valid_decisions_count = 0
                for task in tasks_with_decisions:
                    if task and task.chosen_execution:
                        valid_decisions_count += 1
                        print(
                            f"[TIME {self.env.now:.2f}] Task {task.task_id} → {'CLOUD SERVER' if task.chosen_execution.level==2 else 'EDGE SERVER' if task.chosen_execution.level==1 else 'VEHICLE'} on {task.chosen_execution.platform.name}"
                        )
                    elif task:
                        print(f"[WARNING] Task {task.task_id} has no chosen execution!")

                print(
                    f"[TIME {self.env.now:.2f}] Valid decisions: {valid_decisions_count}/{len([t for t in tasks_with_decisions if t])}"
                )

                return tasks_with_decisions

            except Exception as e:
                print(f"[ERROR] Offloading decision calculation failed: {e}")
                print(f"[TIME {self.env.now:.2f}] Falling back to default decisions...")

                # Fallback: assign local execution for all tasks
                fallback_algo = OffloaingAlgorithme()
                tasks_with_decisions = fallback_algo._fallback_decisions(task_batch)

                yield self.env.timeout(0.1)  # Small delay for fallback processing
                return tasks_with_decisions
        else:
            fallback_algo = OffloaingAlgorithme()
            tasks_with_decisions = fallback_algo._fallback_decisions(task_batch)
            yield self.env.timeout(0.1)  # Small delay for fallback processing
            return tasks_with_decisions

    def _send_decisions_to_vehicles(self, tasks_with_decisions: list[Optional[TaskModelPlatform]]):
        """
        SimPy process to send offloading decisions back to vehicles
        """
        print(f"[TIME {self.env.now:.2f}] Sending offloading decisions to vehicles...")

        # Group tasks by vehicle for efficient communication
        vehicle_task_groups = self._group_tasks_by_vehicle(tasks_with_decisions)

        # Send to each vehicle
        vehicle_processes = []
        for vehicle_id, vehicle_tasks in vehicle_task_groups.items():
            if vehicle_tasks:
                process = self.env.process(
                    self._send_to_specific_vehicle(vehicle_id, vehicle_tasks)
                )
                vehicle_processes.append(process)

        # Wait for all vehicle communications to complete
        if vehicle_processes:
            yield self.env.all_of(vehicle_processes)
            print(f"[TIME {self.env.now:.2f}] All offloading decisions sent to vehicles")
        else:
            print(f"[TIME {self.env.now:.2f}] No valid vehicle communications to send")

    def _group_tasks_by_vehicle(
        self, tasks_with_decisions: list[Optional[TaskModelPlatform]]
    ) -> dict[int, list[TaskModelPlatform]]:
        """
        Group tasks by their source vehicle for efficient communication
        """
        vehicle_groups: dict[int, list[TaskModelPlatform]] = {}

        for task in tasks_with_decisions:
            if task and task.vehicle:
                vehicle_id = task.vehicle.id
                if vehicle_id not in vehicle_groups:
                    vehicle_groups[vehicle_id] = []
                vehicle_groups[vehicle_id].append(task)

        print(
            f"[TIME {self.env.now:.2f}] Grouped {len([t for t in tasks_with_decisions if t])} tasks into {len(vehicle_groups)} vehicle groups"
        )
        return vehicle_groups

    def _send_to_specific_vehicle(self, vehicle_id: int, vehicle_tasks: list[TaskModelPlatform]):
        """
        SimPy process to send offloading decisions to a specific vehicle
        """
        try:
            # Find the vehicle object (this assumes you have access to vehicles)
            target_vehicle = None
            for task in vehicle_tasks:
                if task.vehicle and task.vehicle.id == vehicle_id:
                    target_vehicle = task.vehicle
                    break

            if not target_vehicle:
                print(f"[ERROR] Vehicle {vehicle_id} not found for task communication")
                return

            print(
                f"[TIME {self.env.now:.2f}] Sending {len(vehicle_tasks)} decisions to Vehicle {vehicle_id}"
            )

            # Calculate communication data size (decision metadata)
            decision_data_size = len(vehicle_tasks) * 100  # Assume 100 bytes per decision

            # Simulate network transmission time
            transmission_time = decision_data_size / self.network.edge_to_vehicle_throughput
            transmission_energy = self.power_P_e_v * transmission_time

            print(
                f"[TIME {self.env.now:.2f}] Transmitting {decision_data_size} bytes to Vehicle {vehicle_id} (estimated {transmission_time:.4f}s)"
            )

            # Simulate transmission delay
            yield self.env.timeout(transmission_time)

            # Call vehicle's offloading function
            print(f"[TIME {self.env.now:.2f}] Triggering task execution on Vehicle {vehicle_id}")

            # Start the vehicle's offloading process (non-blocking)
            self.env.process(target_vehicle.offload_task_to_destination(vehicle_tasks))

            print(f"[TIME {self.env.now:.2f}] Successfully sent decisions to Vehicle {vehicle_id}")

        except Exception as e:
            print(f"[ERROR] Failed to send decisions to Vehicle {vehicle_id}: {e}")

    def calculate_response_time(self):
        pass


def calculate_task_response_time(
    # TaskModelPlatform parameters
    offload_level,  # 'local', 'edge', 'cloud'
    T_exec_Mm_rcm,  # Temps d'exécution du modèle Mm sur ressource rcm (donné par votre modèle de coût)
    alpha=0,  # Coefficient alpha (0 si pas de handoff, 1 si handoff)
    # Network parameters
    t_ve_up=0.0,  # Temps de transmission véhicule -> Edge (uplink)
    t_ee_tr=0.0,  # Temps de transmission inter-Edge (si nécessaire pour alpha)
    t_ev_dn=0.0,  # Temps de transmission Edge -> véhicule (downlink)
    t_ec_tr=0.0,  # Temps de transmission Edge -> Cloud
    T_ce_tr=0.0,  # Temps de transmission Cloud -> Edge
):
    """
    Calcule le temps de réponse total (t_resp) pour une tâche,
    selon le niveau de déchargement.

    Args:
        offload_level (str): Le niveau où la tâche est exécutée ('local', 'edge', 'cloud').
        T_exec_Mm_rcm (float): Temps d'exécution du modèle M_m sur la ressource rc_m.
                               Ceci est donné par votre modèle de coût.
        t_ve_up (float): Temps de transmission uplink du véhicule vers l'Edge.
        t_ee_tr (float): Temps de transmission inter-Edge pour le retour des résultats.
        t_ev_dn (float): Temps de transmission downlink de l'Edge vers le véhicule.
        t_ec_tr (float): Temps de transmission de l'Edge vers le Cloud.
        T_ce_tr (float): Temps de transmission du Cloud vers l'Edge.
        alpha (int): Coefficient de handoff (0 si pas de handoff, 1 si handoff).

    Returns:
        float: Le temps de réponse total en secondes.
    """

    t_resp = 0.0

    if offload_level == "local":
        # Temps de réponse de la tâche localement = T_exec_Mm_rcm
        t_resp = T_exec_Mm_rcm

    elif offload_level == "edge":
        # Temps de réponse de la tâche sur e = t_ve_up + T_exec_Mm_rcm + alpha * t_ee_tr + t_ev_dn
        t_resp = t_ve_up + T_exec_Mm_rcm + (alpha * t_ee_tr) + t_ev_dn

    elif offload_level == "cloud":
        # Temps de réponse de la tâche sur c = t_ve_up + t_ec_tr + T_exec_Mm_rcm + T_ce_tr + t_ev_dn
        t_resp = t_ve_up + t_ec_tr + T_exec_Mm_rcm + T_ce_tr + t_ev_dn

    else:
        print(f"Erreur: Niveau de déchargement '{offload_level}' non valide.")
        t_resp = float("inf")  # Représente un temps infini si le niveau n'est pas reconnu

    return t_resp
