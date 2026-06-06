import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Any, Tuple, Optional
import time
import matplotlib.pyplot as plt
from ppo_rl_agent.config import config, norm_params
from ppo_rl_agent.data_cost_model import combinations_cost_model
import ppo_rl_agent.data_generator as data_gen
import copy

# Register the environment
gym.register(id="OffloadingAdEnv-v0", entry_point="ppo_rl_agent.flat_policy:OffloadingAdEnv")

# Constants
PAD_ID = -1
PAD_VALUE_FLOAT = -1.0
PAD_VALUE_INT = -1
# VERY_LARGE_PENALTY = -1


HARD_VIOLATION_PENALTY = -1.0
TASK_COMPLETION_REWARD = 0.5  # Add at top
# UTILIZATION_VIOLATION_PENALTY = -0.3
# MEMORY_VIOLATION_PENALTY = -0.5
# ACCURACY_VIOLATION_PENALTY = -0.1
# LATENCY_VIOLATION_PENALTY = -1
# EDGE_USAGE_BONUS = 0.1  # at top


class OffloadingAdEnv(gym.Env):
    """
    Environment for offloading tasks in an edge-cloud system.
    """

    def __init__(self, config_env: Optional[dict] = None):
        super(OffloadingAdEnv, self).__init__()
        self.current_step = 0
        self.config = config
        self.max_steps = 100  # config.get("max_steps")
        self.N_max = config.get("N_max", 0)
        self.V_max = config.get("V_max", 0)
        self.task_types = config.get("task_types", [])
        self.hardware_names = config.get("hardware_names", [])
        self.models_names = config.get("MODEL_NAMES", [])
        self.levels = config.get("LEVELS", ["Vehicle", "Edge", "Cloud"])
        self.normalization_params: dict = norm_params
        self.combinations = combinations_cost_model
        self.data_generator = data_gen.DataGenerator()
        self.state = None
        self.current_tasks_data: list = []
        self.current_vehicles_data: list = []
        self.current_info_edge: dict = {}
        self.current_info_cloud: dict = {}
        self.current_info_communication: dict = {}
        self.last_step_info = None
        # added in the last version
        # Add these new instance variables
        self.current_time = 0.0
        self.time_step = 0.1  # Seconds per step
        self.task_history = {
            "completed": [],
            "active": [],
            "failed": [],
            "accuracy_failed": 0,
            "task_count": 0,
        }
        self.task_per_platform = {
            "Vehicle": 0,
            "Edge": 0,
            "Cloud": 0,
        }
        self.tasks_to_offload = []
        self.chosen_combinations: dict[str, dict] = {}
        # Define observation space with nested Combinations_Data
        self.observation_space = spaces.Dict(
            {
                "Tasks_Data": spaces.Dict(
                    {
                        "id_Tache": spaces.Box(
                            low=-1, high=self.N_max, shape=(self.N_max, 1), dtype=np.int64
                        ),
                        "Type": spaces.Box(
                            low=-1, high=len(self.task_types), shape=(self.N_max, 1), dtype=np.int32
                        ),
                        "min_accuracy": spaces.Box(
                            low=-1.0, high=1.0, shape=(self.N_max, 1), dtype=np.float32
                        ),
                        "time_to_deadline": spaces.Box(
                            low=-1.0, high=1.0, shape=(self.N_max, 1), dtype=np.float32
                        ),
                        "data_size_input": spaces.Box(
                            low=-1.0, high=1.0, shape=(self.N_max, 1), dtype=np.float32
                        ),
                        "data_size_output": spaces.Box(
                            low=-1.0, high=1.0, shape=(self.N_max, 1), dtype=np.float32
                        ),
                        "alpha_list_edge_comb": spaces.Box(
                            low=0,
                            high=1,
                            shape=(self.N_max, config.get("Max_combinations_per_edge")),
                            dtype=np.int8,
                        ),
                        "vehicle_idx": spaces.Box(
                            low=-1, high=self.V_max, shape=(self.N_max, 1), dtype=np.int64
                        ),
                    }
                ),
                "Vehicles_Data": spaces.Dict(
                    {
                        "id_Vehicule": spaces.Box(
                            low=-1, high=self.V_max, shape=(self.V_max, 1), dtype=np.int64
                        ),
                        "power_v_e": spaces.Box(
                            low=-1.0, high=1.0, shape=(self.V_max, 1), dtype=np.float32
                        ),
                        "hardware_name_id": spaces.Box(
                            low=-1,
                            high=len(self.hardware_names),
                            shape=(self.V_max, len(config.get("Hardware_Vehicle")), 1),
                            dtype=np.int64,
                        ),
                        "charge_remaining_percentage": spaces.Box(
                            low=-1.0,
                            high=1.0,
                            shape=(self.V_max, len(config.get("Hardware_Vehicle")), 1),
                            dtype=np.float32,
                        ),
                        "memory_capacity_remaining_of_this_hardware": spaces.Box(
                            low=-1.0,
                            high=1.0,
                            shape=(self.V_max, len(config.get("Hardware_Vehicle")), 1),
                            dtype=np.float32,
                        ),
                    }
                ),
                "Combinations_Data": spaces.Dict(
                    {
                        task_type: spaces.Dict(
                            {
                                "model_name_id": spaces.Box(
                                    low=-1,
                                    high=len(self.models_names),
                                    shape=(config.get("Max_combinations_per_type"), 1),
                                    dtype=np.int32,
                                ),
                                "hardware_name_id": spaces.Box(
                                    low=-1,
                                    high=len(self.hardware_names),
                                    shape=(config.get("Max_combinations_per_type"), 1),
                                    dtype=np.int32,
                                ),
                                "accuracy": spaces.Box(
                                    low=-1.0,
                                    high=1.0,
                                    shape=(config.get("Max_combinations_per_type"), 1),
                                    dtype=np.float32,
                                ),
                                "execution_time": spaces.Box(
                                    low=-1.0,
                                    high=1.0,
                                    shape=(config.get("Max_combinations_per_type"), 1),
                                    dtype=np.float32,
                                ),
                                "power_consumption": spaces.Box(
                                    low=-1.0,
                                    high=1.0,
                                    shape=(config.get("Max_combinations_per_type"), 1),
                                    dtype=np.float32,
                                ),
                                "memory_consumption": spaces.Box(
                                    low=-1.0,
                                    high=1.0,
                                    shape=(config.get("Max_combinations_per_type"), 1),
                                    dtype=np.float32,
                                ),
                                "utilization_percentage": spaces.Box(
                                    low=-1.0,
                                    high=1.0,
                                    shape=(config.get("Max_combinations_per_type"), 1),
                                    dtype=np.float32,
                                ),
                                "niveau_id": spaces.Box(
                                    low=-1,
                                    high=len(self.levels),
                                    shape=(config.get("Max_combinations_per_type"), 1),
                                    dtype=np.int32,
                                ),
                            }
                        )
                        for task_type in self.task_types
                    }
                ),
                "Info_Edge": spaces.Dict(
                    {
                        "power_P_e_e": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                        "power_P_e_v": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                        "power_P_e_c": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                        "hardware_name_id": spaces.Box(
                            low=-1,
                            high=len(self.hardware_names),
                            shape=(len(config.get("Hardware_Edge")), 1),
                            dtype=np.int64,
                        ),
                        "charge_remaining_percentage": spaces.Box(
                            low=-1.0,
                            high=1.0,
                            shape=(len(config.get("Hardware_Edge")), 1),
                            dtype=np.float32,
                        ),
                        "memory_capacity_remaining_of_this_hardware": spaces.Box(
                            low=-1.0,
                            high=1.0,
                            shape=(len(config.get("Hardware_Edge")), 1),
                            dtype=np.float32,
                        ),
                    }
                ),
                "Info_Cloud": spaces.Dict(
                    {
                        "power_P_c_e": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                        "hardware_name_id": spaces.Box(
                            low=-1,
                            high=len(self.hardware_names),
                            shape=(len(config.get("Hardware_Cloud")), 1),
                            dtype=np.int64,
                        ),
                        "charge_remaining_percentage": spaces.Box(
                            low=-1.0,
                            high=1.0,
                            shape=(len(config.get("Hardware_Cloud")), 1),
                            dtype=np.float32,
                        ),
                        "memory_capacity_remaining_of_this_hardware": spaces.Box(
                            low=-1.0,
                            high=1.0,
                            shape=(len(config.get("Hardware_Cloud")), 1),
                            dtype=np.float32,
                        ),
                    }
                ),
                "Info_Communication": spaces.Dict(
                    {
                        "vehicle_to_edge_throughput": spaces.Box(
                            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                        ),
                        "edge_to_edge_throughput": spaces.Box(
                            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                        ),
                        "edge_to_cloud_throughput": spaces.Box(
                            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                        ),
                        "edge_to_vehicle_throughput": spaces.Box(
                            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                        ),
                        "cloud_to_edge_throughput": spaces.Box(
                            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                        ),
                    }
                ),
            }
        )

        self.action_space = spaces.MultiDiscrete(
            [self.config.get("Max_combinations_per_type")] * self.N_max
        )
        self._episode_counter = 0  # internal episode index

    def normalize(self, value: float, param_key) -> float:
        min_val = self.normalization_params[param_key]["min"]
        max_val = self.normalization_params[param_key]["max"]
        if max_val - min_val == 0:
            return np.array([0.0], dtype=np.float32)
        normalized_value = (value - min_val) / (max_val - min_val)
        return np.array([np.clip(normalized_value, 0.0, 1.0)], dtype=np.float32)

    def get_observation(self) -> Dict[str, Any]:
        null_hardware = {
            "hardware_name_id": np.array(len(self.hardware_names), dtype=np.int64),
            "charge_remaining_percentage": np.array([PAD_VALUE_FLOAT], dtype=np.float32),
            "memory_capacity_remaining_of_this_hardware": np.array(
                [PAD_VALUE_FLOAT], dtype=np.float32
            ),
        }

        # Tasks_Data
        tasks_data = {
            "id_Tache": np.full((self.N_max, 1), PAD_ID, dtype=np.int64),
            "Type": np.full((self.N_max, 1), len(self.task_types), dtype=np.int32),
            "min_accuracy": np.full((self.N_max, 1), PAD_VALUE_FLOAT, dtype=np.float32),
            "time_to_deadline": np.full((self.N_max, 1), PAD_VALUE_FLOAT, dtype=np.float32),
            "data_size_input": np.full((self.N_max, 1), PAD_VALUE_FLOAT, dtype=np.float32),
            "data_size_output": np.full((self.N_max, 1), PAD_VALUE_FLOAT, dtype=np.float32),
            "alpha_list_edge_comb": np.zeros(
                (self.N_max, self.config.get("Max_combinations_per_edge")), dtype=np.int8
            ),
            "vehicle_idx": np.full((self.N_max, 1), PAD_ID, dtype=np.int64),
        }
        for i, task in enumerate(self.tasks_to_offload):
            tasks_data["id_Tache"][i] = task["id_Tache"]
            tasks_data["Type"][i] = task["Type"]
            tasks_data["min_accuracy"][i] = self.normalize(task["min_accuracy"], "min_accuracy")
            tasks_data["time_to_deadline"][i] = self.normalize(
                task["time_to_deadline"], "time_to_deadline"
            )
            tasks_data["data_size_input"][i] = self.normalize(
                task["data_size_input"], "data_size_input"
            )
            tasks_data["data_size_output"][i] = self.normalize(
                task["data_size_output"], "data_size_output"
            )
            tasks_data["alpha_list_edge_comb"][i] = task["alpha_list_edge_comb"]
            tasks_data["vehicle_idx"][i] = task["vehicle_idx"]

        # Vehicles_Data
        vehicles_data = {
            "id_Vehicule": np.full((self.V_max, 1), PAD_ID, dtype=np.int64),
            "power_v_e": np.full((self.V_max, 1), PAD_VALUE_FLOAT, dtype=np.float32),
            "hardware_name_id": np.full(
                (self.V_max, len(config["Hardware_Vehicle"]), 1),
                len(self.hardware_names),
                dtype=np.int64,
            ),
            "charge_remaining_percentage": np.full(
                (self.V_max, len(config["Hardware_Vehicle"]), 1), PAD_VALUE_FLOAT, dtype=np.float32
            ),
            "memory_capacity_remaining_of_this_hardware": np.full(
                (self.V_max, len(config["Hardware_Vehicle"]), 1), PAD_VALUE_FLOAT, dtype=np.float32
            ),
        }
        for i, vehicle in enumerate(self.current_vehicles_data):
            vehicles_data["id_Vehicule"][i] = vehicle["id_Vehicule"]
            vehicles_data["power_v_e"][i] = self.normalize(vehicle["power_v_e"], "power_P_v_e")
            for j, hw in enumerate(vehicle["Hardware_embarques"]):
                vehicles_data["hardware_name_id"][i, j] = hw["hardware_name_id"]
                vehicles_data["charge_remaining_percentage"][i, j] = self.normalize(
                    hw["charge_remaining_percentage"], "charge_remaining_percentage"
                )
                vehicles_data["memory_capacity_remaining_of_this_hardware"][i, j] = self.normalize(
                    hw["memory_capacity_remaining_of_this_hardware"],
                    "memory_capacity_remaining_of_this_hardware",
                )

        # Combinations_Data (nested)
        combinations_data = {}
        for task_type in self.task_types:
            combinations_data[task_type] = {
                "model_name_id": np.full(
                    (self.config["Max_combinations_per_type"], 1), -1, dtype=np.int32
                ),
                "hardware_name_id": np.full(
                    (self.config["Max_combinations_per_type"], 1), -1, dtype=np.int32
                ),
                "accuracy": np.full(
                    (self.config["Max_combinations_per_type"], 1), PAD_VALUE_FLOAT, dtype=np.float32
                ),
                "execution_time": np.full(
                    (self.config["Max_combinations_per_type"], 1), PAD_VALUE_FLOAT, dtype=np.float32
                ),
                "power_consumption": np.full(
                    (self.config["Max_combinations_per_type"], 1), PAD_VALUE_FLOAT, dtype=np.float32
                ),
                "memory_consumption": np.full(
                    (self.config["Max_combinations_per_type"], 1), PAD_VALUE_FLOAT, dtype=np.float32
                ),
                "utilization_percentage": np.full(
                    (self.config["Max_combinations_per_type"], 1), PAD_VALUE_FLOAT, dtype=np.float32
                ),
                "niveau_id": np.full(
                    (self.config["Max_combinations_per_type"], 1), -1, dtype=np.int32
                ),
            }
            for i, comb in enumerate(
                self.combinations.get(task_type, [])[: self.config["Max_combinations_per_type"]]
            ):
                combinations_data[task_type]["model_name_id"][i] = comb["model_name_id"]
                combinations_data[task_type]["hardware_name_id"][i] = comb["hardware_name_id"]
                combinations_data[task_type]["accuracy"][i] = self.normalize(
                    comb["accuracy"], "accuracy_of_model"
                )
                combinations_data[task_type]["execution_time"][i] = self.normalize(
                    comb["execution_time"], "execution_time"
                )
                combinations_data[task_type]["power_consumption"][i] = self.normalize(
                    comb["power_consumption"], "power_consumption"
                )
                combinations_data[task_type]["memory_consumption"][i] = self.normalize(
                    comb["memory_consumption"], "memory_consumption"
                )
                combinations_data[task_type]["utilization_percentage"][i] = self.normalize(
                    comb["utilization_percentage"], "utilization_percentage"
                )
                combinations_data[task_type]["niveau_id"][i] = comb["niveau_id"]

        # Info_Edge
        info_edge = {
            "power_P_e_e": self.normalize(self.current_info_edge["power_P_e_e"], "power_P_e_e"),
            "power_P_e_v": self.normalize(self.current_info_edge["power_P_e_v"], "power_P_e_v"),
            "power_P_e_c": self.normalize(self.current_info_edge["power_P_e_c"], "power_P_e_c"),
            "hardware_name_id": np.full(
                (len(config["Hardware_Edge"]), 1), len(self.hardware_names), dtype=np.int64
            ),
            "charge_remaining_percentage": np.full(
                (len(config["Hardware_Edge"]), 1), PAD_VALUE_FLOAT, dtype=np.float32
            ),
            "memory_capacity_remaining_of_this_hardware": np.full(
                (len(config["Hardware_Edge"]), 1), PAD_VALUE_FLOAT, dtype=np.float32
            ),
        }
        for i, hw in enumerate(self.current_info_edge["Hardware_Edge"]):
            info_edge["hardware_name_id"][i] = hw["hardware_name_id"]
            info_edge["charge_remaining_percentage"][i] = self.normalize(
                hw["charge_remaining_percentage"], "charge_remaining_percentage"
            )
            info_edge["memory_capacity_remaining_of_this_hardware"][i] = self.normalize(
                hw["memory_capacity_remaining_of_this_hardware"],
                "memory_capacity_remaining_of_this_hardware",
            )

        # Info_Cloud
        info_cloud = {
            "power_P_c_e": self.normalize(self.current_info_cloud["power_P_c_e"], "power_P_c_e"),
            "hardware_name_id": np.full(
                (len(config["Hardware_Cloud"]), 1), len(self.hardware_names), dtype=np.int64
            ),
            "charge_remaining_percentage": np.full(
                (len(config["Hardware_Cloud"]), 1), PAD_VALUE_FLOAT, dtype=np.float32
            ),
            "memory_capacity_remaining_of_this_hardware": np.full(
                (len(config["Hardware_Cloud"]), 1), PAD_VALUE_FLOAT, dtype=np.float32
            ),
        }
        for i, hw in enumerate(self.current_info_cloud["Hardware_Cloud"]):
            info_cloud["hardware_name_id"][i] = hw["hardware_name_id"]
            info_cloud["charge_remaining_percentage"][i] = self.normalize(
                hw["charge_remaining_percentage"], "charge_remaining_percentage"
            )
            info_cloud["memory_capacity_remaining_of_this_hardware"][i] = self.normalize(
                hw["memory_capacity_remaining_of_this_hardware"],
                "memory_capacity_remaining_of_this_hardware",
            )

        # Info_Communication
        info_communication = {
            "vehicle_to_edge_throughput": self.normalize(
                self.current_info_communication.get("vehicle_to_edge_throughput", 0.0),
                "vehicle_to_edge_throughput",
            ),
            "edge_to_edge_throughput": self.normalize(
                self.current_info_communication.get("edge_to_edge_throughput", 0.0),
                "edge_to_edge_throughput",
            ),
            "edge_to_cloud_throughput": self.normalize(
                self.current_info_communication.get("edge_to_cloud_throughput", 0.0),
                "edge_to_cloud_throughput",
            ),
            "edge_to_vehicle_throughput": self.normalize(
                self.current_info_communication.get("edge_to_vehicle_throughput", 0.0),
                "edge_to_vehicle_throughput",
            ),
            "cloud_to_edge_throughput": self.normalize(
                self.current_info_communication.get("cloud_to_edge_throughput", 0.0),
                "cloud_to_edge_throughput",
            ),
        }

        return {
            "Tasks_Data": tasks_data,
            "Vehicles_Data": vehicles_data,
            "Combinations_Data": combinations_data,
            "Info_Edge": info_edge,
            "Info_Cloud": info_cloud,
            "Info_Communication": info_communication,
        }

    def reset(
        self, *, seed: Optional[int] = None, options=None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # clearing episode tracking variables
        self.current_tasks_data = []  # Clear current tasks
        self.task_per_platform = {"Vehicle": 0, "Edge": 0, "Cloud": 0}
        self._lat_sum = 0.0
        self._ene_sum = 0.0
        self._acc_sum = 0.0
        self._succ_count = 0

        super().reset(seed=seed)
        (
            self.tasks_to_offload_dict,
            self.current_vehicles_data_dict,
            self.current_info_edge,
            self.current_info_cloud,
            self.current_info_communication,
        ) = self.data_generator.generate_random_state()
        self.tasks_to_offload = list(self.tasks_to_offload_dict.values())
        self.current_vehicles_data = list(self.current_vehicles_data_dict.values())
        # print(f"[INFO] Current Vehicles info {self.current_vehicles_data_dict} vehicles.")
        # print(f"[INFO] Current edge info: {self.current_info_edge}")
        # print(f"[INFO] Current cloud info: {self.current_info_cloud}")
        # Reset tracking variables
        self.current_time = 0.0
        self.current_step = 0
        self.task_history = {
            "completed": [],
            "active": [
                {"task_id": t["id_Tache"], "start_time": 0.0} for t in self.tasks_to_offload
            ],
            "failed": [],
            "accuracy_failed": 0,
            "task_count": 0,
        }

        observation = self.get_observation()  # Keep original observation
        info = {
            "nt_initial_tasks": len(self.tasks_to_offload),
            "vt_initial_vehicles": len(self.current_vehicles_data),
        }
        return observation, info

    def _release_task_resources(self, task):
        """Release resources used by a completed task"""
        # Get the combination used for this task
        default_comb = self.chosen_combinations[task["id_Tache"]]
        comb = task.get("chosen_combination", default_comb)
        if not comb:
            return

        level = self.levels[comb["niveau_id"]]
        hw_id = comb["hardware_name_id"]
        released_mem = comb["memory_consumption"] + task["data_size_input"]
        released_util = comb["utilization_percentage"] if comb["utilization_percentage"] < 1 else 1

        # Find and update the hardware
        vehicle_idx = task["vehicle_idx"]
        if level == "Vehicle":
            if 0 <= vehicle_idx < len(self.current_vehicles_data):
                for hw in self.current_vehicles_data_dict[vehicle_idx]["Hardware_embarques"]:
                    if hw["hardware_name_id"] == hw_id:
                        hw["memory_capacity_remaining_of_this_hardware"] += released_mem
                        hw["charge_remaining_percentage"] += released_util
                        break
            else:
                print(
                    f"[WARNING] Invalid vehicle index {vehicle_idx} for task {task['id_Tache']}, because current vehicles count is {len(self.current_vehicles_data)}"
                )
                return
        elif level == "Edge":
            for hw in self.current_info_edge["Hardware_Edge"]:
                if hw["hardware_name_id"] == hw_id:
                    hw["memory_capacity_remaining_of_this_hardware"] += released_mem
                    hw["charge_remaining_percentage"] += released_util
                    break
        else:  # Cloud
            for hw in self.current_info_cloud["Hardware_Cloud"]:
                if hw["hardware_name_id"] == hw_id:
                    hw["memory_capacity_remaining_of_this_hardware"] += released_mem
                    hw["charge_remaining_percentage"] += released_util
                    break

    def _process_completed_tasks(self):
        """
        Check and remove completed tasks, releasing their resources
        verify is the execution time is over
        if yes, remove it from current_tasks_data
        """
        remaining_tasks = []
        for task in self.current_tasks_data:
            # Update deadline
            task["latency"] -= self.time_step

            if task["latency"] <= 0:
                # print(f"[WARNING] Task {task['id_Tache']} has negative execution_time: {task['latency']}")
                # Task completed (either succeeded or failed)
                self._release_task_resources(task)
            else:
                remaining_tasks.append(task)
        self.current_tasks_data = remaining_tasks

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        # updated by deepseek
        self.current_step += 1
        # print(f"[INFO] NEW STEP : {self.current_step}")
        self.current_time += self.time_step

        # 1. Process completed tasks and release resources
        # print(f"PRE-PROCESSING COMPLETED TASKS : {len(self.current_tasks_data)} current tasks")
        self._process_completed_tasks()

        # 1. Process actions with original reward function
        reward, info = self.reward_function(action)
        # print(f"PRE-REWARD CALCULATION : {len(self.current_tasks_data)} current tasks")
        # 2. Update task deadlines (no reward impact)
        self._update_task_deadlines()
        # print(f"POST-DEADLINE_UPDATE : {len(self.current_tasks_data)} current tasks")
        # print(f"[STEP-INFO] Current Vehicles info {self.current_vehicles_data_dict} vehicles.")
        # print(f"[STEP-INFO] Current edge info: {self.current_info_edge}")
        # print(f"[STEP-INFO] Current cloud info: {self.current_info_cloud}")
        # 3. Generate new tasks while maintaining current resources
        if (
            # (self.np_random.random() > 0.3 or len(self.current_tasks_data)<5)
            (self.current_step / self.max_steps) < 0.9
            and len(self.current_tasks_data) < 50
        ):
            new_tasks_dict = self._generate_new_tasks()
            new_tasks = list(new_tasks_dict.values())
            self.tasks_to_offload = new_tasks
            # print(f"NEW TASKS GENERATED : {len(new_tasks)} tasks")
            # 4. Record new tasks in history
            for task in new_tasks:
                self.task_history["active"].append(
                    {"task_id": task["id_Tache"], "start_time": self.current_time}
                )
        else:
            self.tasks_to_offload = []  # No new tasks this step
        # 5. Prepare next observation (original unchanged)
        next_state = self.get_observation()

        # 6. Check termination (only step count and task completion)
        terminated = len(self.current_tasks_data) == 0 and self.tasks_to_offload == 0
        truncated = self.current_step >= self.max_steps

        if info.get("invalid_action_termination", False) and self.current_step > 200:
            truncated = True

        if terminated or truncated:
            avg_lat = self._lat_sum / self._succ_count if self._succ_count else 0.0
            avg_ene = self._ene_sum / self._succ_count if self._succ_count else 0.0
            avg_acc = self._acc_sum / self._succ_count if self._succ_count else 0.0
            episode_metrics = {
                "success_rate": len(self.task_history["completed"])
                / max(1, self.task_history["task_count"]),
                "avg_latency": avg_lat,
                "avg_energy": avg_ene,
                "avg_accuracy": avg_acc,
                "Vehicles_Number": len(self.current_vehicles_data),
                "Tasks_generated_in_last_step": len(self.tasks_to_offload),
                "Current_Tasks_Number": len(self.current_tasks_data),
                "Edge_tasks_Number": self.task_per_platform["Edge"],
                "Vehicle_tasks_Number": self.task_per_platform["Vehicle"],
                "Cloud_tasks_Number": self.task_per_platform["Cloud"],
            }
            deadline_violation_count = 0
            utilization_violation_count = 0
            memory_violation_count = 0
            accuracy_violation_count = 0
            hardware_unavailable_count = 0
            invalid_decision_count = 0
            missing_action_count = 0
            for t in self.task_history["failed"]:
                if t.get("reason") == "deadline_violation":
                    deadline_violation_count += 1
                elif t.get("reason") == "utilization_violation":
                    utilization_violation_count += 1
                elif t.get("reason") == "memory_violation":
                    memory_violation_count += 1
                elif t.get("reason") == "hardware_unavailable":
                    hardware_unavailable_count += 1
                elif t.get("reason") == "invalid_decision":
                    invalid_decision_count += 1
                else:
                    missing_action_count += 1

            info["Reason_for_tasks_failed"] = {
                "deadline_violation": deadline_violation_count,
                "utilization_violation": utilization_violation_count,
                "memory_violation": memory_violation_count,
                "accuracy_violation": self.task_history["accuracy_failed"],
                "hardware_unavailable": hardware_unavailable_count,
                "invalid_decision": invalid_decision_count,
                "missing_action": missing_action_count,
            }
            info.update(episode_metrics)

            # ready for next episode
            self._episode_counter += 1
        # 7. Add history info for monitoring
        info.update(
            {
                "current_time": self.current_time,
                "active_tasks": len(self.task_history["active"]),
                "completed_tasks": len(self.task_history["completed"]),
                "failed_tasks": self.task_history["task_count"]
                - len(self.task_history["completed"]),
            }
        )

        return next_state, reward, terminated, truncated, info

    def _update_task_deadlines(self):
        """Internal: Update deadlines and move expired tasks to failed"""
        active_tasks = []
        for task in self.current_tasks_data:
            task["time_to_deadline"] -= self.time_step

            if task["time_to_deadline"] > 0:
                active_tasks.append(task)
            else:
                self._record_task_outcome(
                    task["id_Tache"], success=False, failure_reason="deadline_expired"
                )
                self._release_task_resources(task)
        self.current_tasks_data = active_tasks

    def _generate_new_tasks(self) -> Dict[int, Dict[str, Any]]:
        """Internal: Generate new tasks while preserving current vehicles"""
        V_t = len(self.current_vehicles_data)
        N_t = min(self.np_random.integers(1, self.N_max + 1), V_t * len(self.task_types))
        return self.data_generator.generate_tasks(N_t, V_t)

    def _record_task_outcome(
        self, task_id: int, success: bool, failure_reason: Optional[str] = None
    ):
        """Internal: Track task completion/failure"""
        # record = {"task_id": task_id, "time": self.current_time}
        record = {}
        if not success:
            if failure_reason == "accuracy_violation":
                self.task_history["accuracy_failed"] += 1
            else:
                record["reason"] = failure_reason
                self.task_history["failed"].append(record)
        else:
            self.task_history["completed"].append(record)

        # Remove from active
        self.task_history["active"] = [
            t for t in self.task_history["active"] if t["task_id"] != task_id
        ]

    def reward_function(self, action: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        reward = 0.0
        info: dict = {}
        total_latency_all_tasks = 0.0
        total_energy_all_tasks = 0.0
        processed_task_ids = []

        temp_hardware_capacities: Dict[Tuple[str, int], Dict[str, float]] = {}
        if len(self.tasks_to_offload) == 0:
            # reward = -0.1 * len(self.current_tasks_data) / self.N_max
            return reward, info

        for veh in self.current_vehicles_data:
            for hw in veh["Hardware_embarques"]:
                key = (f"Vehicle_{veh['id_Vehicule']}", hw["hardware_name_id"])
                temp_hardware_capacities[key] = copy.deepcopy(
                    {
                        "charge_remaining_percentage": hw["charge_remaining_percentage"],
                        "memory_capacity_remaining_of_this_hardware": hw[
                            "memory_capacity_remaining_of_this_hardware"
                        ],
                    }
                )

        for hw in self.current_info_edge["Hardware_Edge"]:
            key = ("Edge", hw["hardware_name_id"])
            temp_hardware_capacities[key] = copy.deepcopy(
                {
                    "charge_remaining_percentage": hw["charge_remaining_percentage"],
                    "memory_capacity_remaining_of_this_hardware": hw[
                        "memory_capacity_remaining_of_this_hardware"
                    ],
                }
            )

        for hw in self.current_info_cloud["Hardware_Cloud"]:
            key = ("Cloud", hw["hardware_name_id"])
            temp_hardware_capacities[key] = copy.deepcopy(
                {
                    "charge_remaining_percentage": hw["charge_remaining_percentage"],
                    "memory_capacity_remaining_of_this_hardware": hw[
                        "memory_capacity_remaining_of_this_hardware"
                    ],
                }
            )

        for i, task in enumerate(self.tasks_to_offload):
            task_id = task["id_Tache"]
            task_failed = False  # Explicit initialization
            task_type_idx = task["Type"]
            task_input_data_size = task["data_size_input"]
            self.task_history["task_count"] += 1

            if i >= len(action):
                reward += -1
                failure_reason = "missing_action"
                self._record_task_outcome(task_id, False, failure_reason)
                continue

            chosen_combination_idx = action[i]
            task_type_str = self._get_task_type_string(task_type_idx)

            if task_type_str == "UNKNOWN_TYPE" or task_id == PAD_ID:
                continue

            available_combinations = self.combinations.get(task_type_str, [])

            if chosen_combination_idx >= len(available_combinations):
                reward += -1
                failure_reason = "invalid_decision"
                self._record_task_outcome(task_id, False, failure_reason)
                continue

            chosen_combination = available_combinations[chosen_combination_idx].copy()

            execution_time = chosen_combination["execution_time"]
            power_consumption_model = chosen_combination["power_consumption"]
            memory_consumption_model = chosen_combination["memory_consumption"]
            accuracy = chosen_combination["accuracy"]
            level = self.levels[chosen_combination["niveau_id"]]
            hardware_id: int = chosen_combination["hardware_name_id"]
            utilization_percentage_model = chosen_combination["utilization_percentage"]
            # track # of decisions per level
            self.task_per_platform[level] += 1
            platform_key_for_lookup = ""
            if level == "Vehicle":
                platform_key_for_lookup = f"Vehicle_{task['vehicle_idx']}"
            elif level == "Edge":
                platform_key_for_lookup = "Edge"
            elif level == "Cloud":
                platform_key_for_lookup = "Cloud"

            hw_resource_key = (platform_key_for_lookup, hardware_id)
            # Ensure the hardware exists in its corresponding offloading decision level
            if hw_resource_key not in temp_hardware_capacities:
                reward += -1
                task_failed = True
                failure_reason = "hardware_unavailable"
                self._record_task_outcome(task_id, False, failure_reason)

            target_hw_capacities = temp_hardware_capacities[hw_resource_key]
            # Verifying Hardware utilization violation
            # if i<= 3 or i >= len(self.tasks_to_offload) - 3:
            #     print(f"AVAILABLE PERCENTAGE of hardware : {hardware_id} of level: {level} is {target_hw_capacities['charge_remaining_percentage']}")
            #     print(f"UTILISATION PERCENTAGE of model :{chosen_combination['model_name_id']} is {utilization_percentage_model}")
            if (
                utilization_percentage_model > target_hw_capacities["charge_remaining_percentage"]
            ) or utilization_percentage_model > 1:
                info["invalid_action_termination"] = True
                reward += -10
                task_failed = True
                failure_reason = "utilization_violation"
                self._record_task_outcome(task_id, False, failure_reason)

            # if i<= 3 or i >= len(self.tasks_to_offload) - 3:
            #     print(f"REAL AVAILABLE MEMORY of hardware : {hardware_id} of level: {level} is {target_hw_capacities['memory_capacity_remaining_of_this_hardware']}")
            #     print(f"MEMORY CONSUMPTION of model :{chosen_combination['model_name_id']} is {memory_consumption_model + task_input_data_size}")
            if (
                memory_consumption_model + task_input_data_size
                > target_hw_capacities["memory_capacity_remaining_of_this_hardware"]
            ):
                reward += -1
                task_failed = True
                failure_reason = "memory_violation"
                self._record_task_outcome(task_id, False, failure_reason)

            task_latency = execution_time
            task_energy = power_consumption_model * execution_time
            comm_info = self.current_info_communication

            if level == "Vehicle":
                pass
            elif level == "Edge":
                t_ve_up = (
                    task["data_size_input"] / comm_info["vehicle_to_edge_throughput"]
                    if comm_info["vehicle_to_edge_throughput"] > 0
                    else float("inf")
                )
                task_latency += t_ve_up
                task_energy += (
                    self.current_vehicles_data_dict[task["vehicle_idx"]]["power_v_e"] * t_ve_up
                )
                edge_comb_list_for_type = [
                    c for c in available_combinations if self.levels[c["niveau_id"]] == "Edge"
                ]
                chosen_edge_comb_internal_idx = -1
                for idx, ec in enumerate(edge_comb_list_for_type):
                    if (
                        ec["model_name_id"] == chosen_combination["model_name_id"]
                        and ec["hardware_name_id"] == chosen_combination["hardware_name_id"]
                    ):
                        chosen_edge_comb_internal_idx = idx
                        break
                if chosen_edge_comb_internal_idx != -1 and chosen_edge_comb_internal_idx < len(
                    task["alpha_list_edge_comb"]
                ):
                    alpha_val = task["alpha_list_edge_comb"][chosen_edge_comb_internal_idx]
                    if alpha_val == 1:
                        t_ee_tr = (
                            task["data_size_output"] / comm_info["edge_to_edge_throughput"]
                            if comm_info["edge_to_edge_throughput"] > 0
                            else float("inf")
                        )
                        task_latency += t_ee_tr
                        task_energy += self.current_info_edge["power_P_e_e"] * t_ee_tr
                t_ev_dn = (
                    task["data_size_output"] / comm_info["edge_to_vehicle_throughput"]
                    if comm_info["edge_to_vehicle_throughput"] > 0
                    else float("inf")
                )
                task_latency += t_ev_dn
                task_energy += self.current_info_edge["power_P_e_v"] * t_ev_dn
            elif level == "Cloud":
                t_ve_up = (
                    task["data_size_input"] / comm_info["vehicle_to_edge_throughput"]
                    if comm_info["vehicle_to_edge_throughput"] > 0
                    else float("inf")
                )
                task_latency += t_ve_up
                task_energy += (
                    self.current_vehicles_data_dict[task["vehicle_idx"]]["power_v_e"] * t_ve_up
                )
                t_ec_tr = (
                    task["data_size_input"] / comm_info["edge_to_cloud_throughput"]
                    if comm_info["edge_to_cloud_throughput"] > 0
                    else float("inf")
                )
                task_latency += t_ec_tr
                task_energy += self.current_info_edge["power_P_e_c"] * t_ec_tr
                t_ce_tr = (
                    task["data_size_output"] / comm_info["cloud_to_edge_throughput"]
                    if comm_info["cloud_to_edge_throughput"] > 0
                    else float("inf")
                )
                task_latency += t_ce_tr
                task_energy += self.current_info_cloud["power_P_c_e"] * t_ce_tr
                t_ev_dn = (
                    task["data_size_output"] / comm_info["edge_to_vehicle_throughput"]
                    if comm_info["edge_to_vehicle_throughput"] > 0
                    else float("inf")
                )
                task_latency += t_ev_dn
                task_energy += self.current_info_edge["power_P_e_v"] * t_ev_dn

            if task_latency > task["time_to_deadline"]:
                reward += -2
                task_failed = True
                failure_reason = "deadline_violation"
                self._record_task_outcome(task_id, False, failure_reason)
            if accuracy < task["min_accuracy"]:
                reward += (
                    -accuracy / task["min_accuracy"]
                )  # negative reward (the far it is than the min accuracy the bigget the negative reward)
                failure_reason = "accuracy_violation"
                self._record_task_outcome(task_id, False, failure_reason)

            if task_failed:
                reward += -1  # failure reward
                continue

            # If task succeeded, update the actual hardware resources
            if level == "Vehicle":
                reward += 1
                for hw in self.current_vehicles_data_dict[task["vehicle_idx"]][
                    "Hardware_embarques"
                ]:
                    if hw["hardware_name_id"] == hardware_id:
                        target_hw_capacities[
                            "charge_remaining_percentage"
                        ] -= utilization_percentage_model
                        target_hw_capacities["memory_capacity_remaining_of_this_hardware"] -= (
                            memory_consumption_model + task_input_data_size
                        )
                        hw["charge_remaining_percentage"] -= utilization_percentage_model
                        hw["memory_capacity_remaining_of_this_hardware"] -= (
                            memory_consumption_model + task_input_data_size
                        )
                        break
            elif level == "Edge":
                for hw in self.current_info_edge["Hardware_Edge"]:
                    if hw["hardware_name_id"] == hardware_id:
                        target_hw_capacities[
                            "charge_remaining_percentage"
                        ] -= utilization_percentage_model
                        target_hw_capacities["memory_capacity_remaining_of_this_hardware"] -= (
                            memory_consumption_model + task_input_data_size
                        )
                        hw["charge_remaining_percentage"] -= utilization_percentage_model
                        hw["memory_capacity_remaining_of_this_hardware"] -= (
                            memory_consumption_model + task_input_data_size
                        )
                        break
            elif level == "Cloud":
                for hw in self.current_info_cloud["Hardware_Cloud"]:
                    if hw["hardware_name_id"] == hardware_id:
                        target_hw_capacities[
                            "charge_remaining_percentage"
                        ] -= utilization_percentage_model
                        target_hw_capacities["memory_capacity_remaining_of_this_hardware"] -= (
                            memory_consumption_model + task_input_data_size
                        )
                        hw["charge_remaining_percentage"] -= utilization_percentage_model
                        hw["memory_capacity_remaining_of_this_hardware"] -= (
                            memory_consumption_model + task_input_data_size
                        )
                        break
            task_in_progress = copy.deepcopy(task)
            # add the chosen combination to the task data
            task_in_progress["chosen_combination"] = chosen_combination.copy()
            task_in_progress["latency"] = task_latency
            self.current_tasks_data.append(task_in_progress)
            self._record_task_outcome(task_id, True)
            total_latency_all_tasks += task_latency
            total_energy_all_tasks += task_energy
            processed_task_ids.append(task_id)
            self.chosen_combinations[task_id] = (
                chosen_combination.copy()
            )  # Track chosen combination
            # updating episode tracking variables
            self._lat_sum += task_latency
            self._ene_sum += task_energy
            self._acc_sum += accuracy
            self._succ_count += 1

        reward += self._succ_count
        # penalty for total latency and energy (objective to be minimized)
        reward += -(total_latency_all_tasks + total_energy_all_tasks)
        if (
            len(processed_task_ids) < len(self.current_tasks_data)
            and len(self.current_tasks_data) > 0
        ):
            unfinished_count = len(self.current_tasks_data) - len(processed_task_ids)
            reward += -unfinished_count / len(self.current_tasks_data)
        # For example, reward for balanced usage (encourage using all platforms)
        # used_platforms = sum(1 for count in self.task_per_platform.values() if count > 0)
        # balance_reward = (2 - used_platforms) / 3  # Normalize to [-1, 1]
        # reward += balance_reward # if one platform, negative penality, if 2 nothing, if 3 positive reward
        return reward if isinstance(reward, float) else reward[0], info

    def _get_task_type_string(self, task_type_idx: int) -> str:
        if task_type_idx < 0 or task_type_idx >= len(self.task_types):
            return "UNKNOWN_TYPE"
        return self.task_types[task_type_idx]

    def render(self, mode="human") -> None:
        if mode != "human":
            raise NotImplementedError(
                f"Render mode '{mode}' is not supported. Only 'human' is supported."
            )
        plt.clf()
        print("\n=== Environment State ===")
        print(f"Current Step: {self.current_step}/{self.max_steps-1}")
        print(f"Number of Active Tasks: {len(self.current_tasks_data)}")
        print(f"Number of Active Vehicles: {len(self.current_vehicles_data)}")
        print("\nTasks:")
        for task in self.current_tasks_data:
            task_id = task["id_Tache"]
            task_type = self._get_task_type_string(task["Type"])
            print(
                f"  Task ID: {task_id}, Type: {task_type}, "
                f"Min Accuracy: {task['min_accuracy']:.2f}, "
                f"Time to Deadline: {task['time_to_deadline']:.2f}s, "
                f"Data Size (In/Out): {task['data_size_input']:.2f}/{task['data_size_output']:.2f} MB",
                f"Vehicle Index: {task['vehicle_idx']}",
                f"Alpha List: {task['alpha_list_edge_comb']}",
            )
        print("\nVehicles:")
        for vehicle in self.current_vehicles_data:
            veh_id = vehicle["id_Vehicule"]
            print(f"  Vehicle ID: {veh_id}, Power: {vehicle['power_v_e'].item():.2f} W")
            for hw in vehicle["Hardware_embarques"]:
                hw_name = (
                    self.hardware_names[hw["hardware_name_id"]]
                    if hw["hardware_name_id"] != PAD_ID
                    else "PAD"
                )
                print(
                    f"    Hardware: {hw_name}, Charge: {hw['charge_remaining_percentage'].item():.2f}%, "
                    f"Memory: {hw['memory_capacity_remaining_of_this_hardware'].item():.2f} GB"
                )
        print("\nEdge Hardware:")
        for hw in self.current_info_edge["Hardware_Edge"]:
            hw_name = (
                self.hardware_names[hw["hardware_name_id"]]
                if hw["hardware_name_id"] != PAD_ID
                else "PAD"
            )
            print(
                f"  Hardware: {hw_name}, Charge: {hw['charge_remaining_percentage'].item():.2f}%, "
                f"Memory: {hw['memory_capacity_remaining_of_this_hardware'].item():.2f} GB"
            )
        print("\nCloud Hardware:")
        for hw in self.current_info_cloud["Hardware_Cloud"]:
            hw_name = (
                self.hardware_names[hw["hardware_name_id"]]
                if hw["hardware_name_id"] != PAD_ID
                else "PAD"
            )
            print(
                f"  Hardware: {hw_name}, Charge: {hw['charge_remaining_percentage'].item():.2f}%, "
                f"Memory: {hw['memory_capacity_remaining_of_this_hardware'].item():.2f} GB"
            )
        task_ids = [task["id_Tache"] for task in self.current_tasks_data]
        if hasattr(self, "last_step_info") and self.last_step_info:
            latencies = [self.last_step_info.get(f"task_{tid}_latency", 0.0) for tid in task_ids]
            energies = [self.last_step_info.get(f"task_{tid}_energy", 0.0) for tid in task_ids]
        else:
            latencies = [0.0] * len(task_ids)
            energies = [0.0] * len(task_ids)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.bar(task_ids, latencies, color="blue", alpha=0.6)
        ax1.set_title("Task Latencies")
        ax1.set_xlabel("Task ID")
        ax1.set_ylabel("Latency (s)")
        ax1.grid(True, alpha=0.3)
        ax2.bar(task_ids, energies, color="green", alpha=0.6)
        ax2.set_title("Task Energy Consumption")
        ax2.set_xlabel("Task ID")
        ax2.set_ylabel("Energy (J)")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.pause(0.1)
        print("\n======================\n")

    def close(self) -> None:
        try:
            plt.close("all")
        except Exception as e:
            print(f"Warning: Error closing matplotlib figures: {e}")
        self.current_tasks_data = None
        self.current_vehicles_data = None
        self.current_info_edge = None
        self.current_info_cloud = None
        self.current_info_communication = None
        self.last_step_info = None
        if hasattr(self.data_generator, "close"):
            try:
                self.data_generator.close()
            except Exception as e:
                print(f"Warning: Error closing DataGenerator: {e}")
        print("Environment closed successfully.")
