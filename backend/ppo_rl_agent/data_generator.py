import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ppo_rl_agent.config import config, norm_params

norm_params: dict[str, dict] = norm_params

task_types = config.get("task_types", [])
NUM_TASK_TYPES = len(task_types)
hardware_names = config.get("hardware_names", [])
NUM_HARDWARE_TYPES = len(hardware_names)
HARDWARE_NAME_TO_ID = {name: i for i, name in enumerate(hardware_names)}

# Task type templates with realistic data size ranges
# Data sizes derived from typical requirements for each task type
TASK_TEMPLATES = {
    0: {  # DO - Object Detection - medium to large images/video frames
        "data_size_range": (0.5, 5),  # MB
        "min_accuracy": 0.65,  # From SSD-MobileNetV3 on INTEL_I7
        "max_accuracy": 0.95,  # From multiple cloud models
    },
    1: {  # CI - Classification - smaller data sizes
        "data_size_range": (0.2, 1),  # MB
        "min_accuracy": 0.70,  # From MobileNetV3-Small
        "max_accuracy": 0.95,  # From ViT-Base
    },
    2: {  # S - Segmentation - large data for pixel-level processing
        "data_size_range": (0.5, 5),  # MB
        "min_accuracy": 0.65,  # From MobileUnet on NANO
        "max_accuracy": 0.95,  # From multiple cloud models
    },
    3: {  # OT - Object Tracking - variable sizes for sequences
        "data_size_range": (0.5, 5),  # MB
        "min_accuracy": 0.65,  # From DeepSORT on TX2
        "max_accuracy": 0.95,  # From multiple cloud models
    },
    4: {  # TLD - Traffic Light Detection - smaller, specific sizes
        "data_size_range": (0.5, 5),  # MB
        "min_accuracy": 0.60,  # From YOLOv3-Tiny-TLD on NANO
        "max_accuracy": 0.95,  # From multiple cloud models
    },
}

# Destination-based attribute ranges
DESTINATION_CONFIGS = {
    "vehicle": {
        "accuracy_range": (0.65, 0.8),  # Lower accuracy requirements
        "deadline_range": (0.05, 0.2),  # Tighter deadlines (seconds)
    },
    "edge": {
        "accuracy_range": (0.75, 0.95),  # Medium accuracy requirements
        "deadline_range": (0.8, 2.5),  # Medium deadlines
    },
    "cloud": {
        "accuracy_range": (0.80, 0.95),  # Higher accuracy requirements
        "deadline_range": (1.0, 3.0),  # More relaxed deadlines
    },
    "random": {
        "accuracy_range": (0.65, 0.95),  # Full range for complex tasks
        "deadline_range": (0.05, 3.0),  # Full deadline range
    },
}


class DataGenerator:
    def __init__(self):
        self.N_max_obs = 5  # config.get("N_max")
        self.V_max_obs = 1  # config.get("V_max")
        self.np_random = np.random.default_rng()

    # def set_seed(self, seed: Optional[int] = None):
    #     """Sets the seed for reproducibility."""
    #     self.np_random = np.random.default_rng(seed)

    def generate_tasks(self, N_t: int, V_t: int) -> dict[int, dict[str, Any]]:
        """
        Generate N_t tasks with enhanced logic for realistic attributes:
        - Task attributes based on task type templates and intended destinations
        - 4 destination types: vehicle (25%), edge (25%), cloud (25%), random (25%)
        - Data sizes depend on task type, accuracy/deadlines depend on destination
        - Each task belongs to a valid vehicle (0 to V_t-1)
        - No vehicle has multiple tasks of the same type

        Args:
            N_t: Number of tasks to generate.
            V_t: Number of vehicles available.

        Returns:
            Dictionary mapping task IDs to task data.
        """
        if V_t > self.V_max_obs:
            raise ValueError(f"V_t ({V_t}) cannot exceed V_max ({self.V_max_obs})")

        tasks_data: dict[int, dict[str, Any]] = {}

        # Cap N_t to ensure constraint feasibility (max V_t tasks per type)
        N_t = min(N_t, V_t * NUM_TASK_TYPES)

        # Track task types per vehicle
        vehicle_task_types: dict[int, set[int]] = {v: set() for v in range(V_t)}

        # Destination types for equal distribution (25% each)
        destination_types = ["vehicle", "edge", "cloud", "random"]

        # Generate tasks
        task_id = 0
        retries = 0
        max_retries = 100

        while task_id < N_t and retries < max_retries:
            # Random task type
            task_type_idx = self.np_random.integers(0, NUM_TASK_TYPES)

            # Find available vehicles
            available_vehicles = [
                v for v in range(V_t) if task_type_idx not in vehicle_task_types[v]
            ]

            if not available_vehicles:
                retries += 1
                continue

            # Randomly select a vehicle
            vehicle_idx = self.np_random.choice(available_vehicles)
            if vehicle_idx >= V_t:  # Double-check
                print(f"[WARNING] Generated invalid vehicle_idx {vehicle_idx} for V_t {V_t}")
                retries += 1
                continue

            # Select destination type (25% each)
            destination_type = self.np_random.choice(destination_types)

            # Get task template and destination config
            task_type_template = TASK_TEMPLATES[task_type_idx]
            destined_level_template = DESTINATION_CONFIGS[destination_type]

            # Generate data size based on task type template
            data_size_input = self.np_random.integers(*task_type_template["data_size_range"])

            # Generate accuracy based on destination and task type constraints
            min_acc = max(
                destined_level_template["accuracy_range"][0], task_type_template["min_accuracy"]
            )
            max_acc = min(
                destined_level_template["accuracy_range"][1], task_type_template["max_accuracy"]
            )
            min_accuracy = self.np_random.uniform(min_acc, max_acc)

            # Generate deadline based on destination
            time_to_deadline = self.np_random.uniform(*destined_level_template["deadline_range"])

            # Generate task attributes
            task = {
                "id_Tache": task_id,
                "Type": task_type_idx,
                "min_accuracy": min_accuracy,
                "time_to_deadline": time_to_deadline,
                "data_size_input": data_size_input,
                "data_size_output": self.np_random.integers(0.1, 5),  # Keep original logic
                "alpha_list_edge_comb": self.np_random.choice(
                    [0, 1],
                    size=config.get("Max_combinations_per_edge"),
                    p=[0.8, 0.2],  # 80% chance of 0, 20% chance of 1
                ).tolist(),
                "vehicle_idx": vehicle_idx,
            }

            # Add task and update tracking
            tasks_data[task_id] = task
            vehicle_task_types[vehicle_idx].add(task_type_idx)
            task_id += 1
            retries = 0

        if task_id < N_t:
            print(
                f"Warning: Generated {task_id} tasks instead of {N_t} due to vehicle-type constraints"
            )

        return tasks_data

    def generate_vehicles(self, V_t: int) -> dict[int, dict[str, Any]]:
        """
        Generate V_t vehicles with random attributes, coherent with vehicle IDs used in generate_tasks.
        Each vehicle has exactly one NANO and one Intel_i7 hardware (no padding).

        Args:
            V_t: Number of vehicles to generate.

        Returns:
            Dictionary mapping vehicle IDs to vehicle data.
        """
        if V_t > self.V_max_obs:
            raise ValueError(f"V_t ({V_t}) cannot exceed V_max ({self.V_max_obs})")

        vehicles_data: dict[int, dict[str, Any]] = {}

        for v_id in range(V_t):
            # Generate hardware entries (exactly one NANO and one Intel_i7)
            hardware_list = []
            hardware_ids = [HARDWARE_NAME_TO_ID[hw] for hw in config["Hardware_Vehicle"]]
            self.np_random.shuffle(hardware_ids)

            for hw_id in hardware_ids:
                charge = 1.0
                memory = self.np_random.uniform(
                    norm_params["memory_capacity_remaining_of_this_hardware"]["max"] * 0.1,
                    norm_params["memory_capacity_remaining_of_this_hardware"]["max"] * 0.125,
                )

                hardware_list.append(
                    {
                        "hardware_name_id": hw_id,
                        "charge_remaining_percentage": np.array(charge, dtype=np.float32),
                        "memory_capacity_remaining_of_this_hardware": np.array(
                            memory, dtype=np.float32
                        ),
                    }
                )

            # Generate vehicle attributes
            vehicle = {
                "id_Vehicule": v_id,
                "power_v_e": np.array(
                    self.np_random.uniform(
                        norm_params["power_P_v_e"]["min"], norm_params["power_P_v_e"]["max"] * 0.2
                    ),
                    dtype=np.float32,
                ),
                "Hardware_embarques": tuple(hardware_list),
            }

            vehicles_data[v_id] = vehicle

        return vehicles_data

    def generate_edges(self) -> dict[str, Any]:
        """
        Generate a single edge node with random attributes.

        Returns:
            Dictionary containing edge node data.
        """
        # Generate hardware entries (3 per edge, per config["Hardware_Edge"])
        hardware_list = []
        hardware_ids = [HARDWARE_NAME_TO_ID[hw] for hw in config["Hardware_Edge"]]
        self.np_random.shuffle(hardware_ids)

        for hw_id in hardware_ids:
            charge = 1
            memory = self.np_random.uniform(
                norm_params["memory_capacity_remaining_of_this_hardware"]["max"] * 0.4,
                norm_params["memory_capacity_remaining_of_this_hardware"]["max"] * 0.5,
            )

            hardware_list.append(
                {
                    "hardware_name_id": hw_id,
                    "charge_remaining_percentage": np.array(charge, dtype=np.float32),
                    "memory_capacity_remaining_of_this_hardware": np.array(
                        memory, dtype=np.float32
                    ),
                }
            )

        # Generate edge attributes
        edge_data = {
            "power_P_e_e": np.array(
                self.np_random.uniform(
                    norm_params["power_P_e_e"]["min"], norm_params["power_P_e_e"]["max"] * 0.15
                ),
                dtype=np.float32,
            ),
            "power_P_e_v": np.array(
                self.np_random.uniform(
                    norm_params["power_P_e_v"]["min"], norm_params["power_P_e_v"]["max"] * 0.15
                ),
                dtype=np.float32,
            ),
            "power_P_e_c": np.array(
                self.np_random.uniform(
                    norm_params["power_P_e_c"]["min"], norm_params["power_P_e_c"]["max"] * 0.4
                ),
                dtype=np.float32,
            ),
            "Hardware_Edge": tuple(hardware_list),
        }

        return edge_data

    def generate_cloud(self) -> dict[str, Any]:
        """
        Generate a single cloud node with random attributes.

        Returns:
            Dictionary containing cloud node data.
        """
        # Generate hardware entries (4 per cloud, per config["Hardware_Cloud"])
        hardware_list = []
        hardware_ids = [HARDWARE_NAME_TO_ID[hw] for hw in config["Hardware_Cloud"]]
        self.np_random.shuffle(hardware_ids)

        for hw_id in hardware_ids:
            charge = 1
            memory = self.np_random.uniform(
                norm_params["memory_capacity_remaining_of_this_hardware"]["max"] * 0.8,
                norm_params["memory_capacity_remaining_of_this_hardware"]["max"],
            )

            hardware_list.append(
                {
                    "hardware_name_id": hw_id,
                    "charge_remaining_percentage": np.array(charge, dtype=np.float32),
                    "memory_capacity_remaining_of_this_hardware": np.array(
                        memory, dtype=np.float32
                    ),
                }
            )

        # Generate cloud attributes
        cloud_data = {
            "power_P_c_e": np.array(
                self.np_random.uniform(
                    norm_params["power_P_c_e"]["min"], norm_params["power_P_c_e"]["max"] * 0.5
                ),
                dtype=np.float32,
            ),
            "Hardware_Cloud": tuple(hardware_list),
        }

        return cloud_data

    def generate_network(self) -> dict[str, float]:
        """
        Generate network communication attributes.

        Returns:
            Dictionary mapping throughput types to float values.
        """
        network_data = {
            "vehicle_to_edge_throughput": float(
                self.np_random.uniform(
                    norm_params["vehicle_to_edge_throughput"]["min"] + 70,
                    norm_params["vehicle_to_edge_throughput"]["max"] * 0.15,
                )
            ),
            "edge_to_edge_throughput": float(
                self.np_random.uniform(
                    norm_params["edge_to_edge_throughput"]["min"] + 500,
                    norm_params["edge_to_edge_throughput"]["max"],
                )
            ),
            "edge_to_cloud_throughput": float(
                self.np_random.uniform(
                    norm_params["edge_to_cloud_throughput"]["min"] + 200,
                    norm_params["edge_to_cloud_throughput"]["max"] * 0.5,
                )
            ),
            "edge_to_vehicle_throughput": float(
                self.np_random.uniform(
                    norm_params["edge_to_vehicle_throughput"]["min"] + 400,
                    norm_params["edge_to_vehicle_throughput"]["max"] * 0.7,
                )
            ),
            "cloud_to_edge_throughput": float(
                self.np_random.uniform(
                    norm_params["cloud_to_edge_throughput"]["min"] + 500,
                    norm_params["cloud_to_edge_throughput"]["max"] * 0.7,
                )
            ),
        }

        return network_data

    def generate_random_state(self) -> Tuple[
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
    ]:
        """
        Generate a random state for the environment.

        Returns:
            Tuple of tasks, vehicles, edge, cloud, and network data.
        """
        V_t = self.V_max_obs  # self.np_random.integers(1, self.V_max_obs + 1)
        N_t = self.np_random.integers(V_t, min(self.N_max_obs + 1, V_t * NUM_TASK_TYPES + 1))
        # print(f"Generating {N_t} tasks for {V_t} vehicles...")

        return (
            self.generate_tasks(N_t, V_t),
            self.generate_vehicles(V_t),
            self.generate_edges(),
            self.generate_cloud(),
            self.generate_network(),
        )
