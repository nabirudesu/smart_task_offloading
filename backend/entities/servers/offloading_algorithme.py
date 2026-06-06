import os
import time
import json
from typing import Optional, Any
import ray

from entities.task_model_platform import TaskModelPlatform
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import numpy as np
import pickle
from ppo_rl_agent.flat_env import OffloadingAdEnv
from ppo_rl_agent.flat_policy import CustomTFModel
import warnings
from ray.tune.logger import NoopLogger
import random

TIME_SLOT = 10
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
LEVELS = ["Vehicle", "Edge", "Cloud"]

# Suppress Gymnasium warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")


def register_custom_components():
    """Register all custom components with RLlib."""

    def env_creator(env_config):
        return OffloadingAdEnv()

    register_env("OffloadingAdEnv-v0", env_creator)
    ModelCatalog.register_custom_model("CustomTFModel", CustomTFModel)


class OffloaingAlgorithme:
    def __init__(self):
        """
        Initialize the offloading algorithm.
        """
        self.model = None

    def _assign_chosen_executions(
        self,
        task_batch: list[Optional[TaskModelPlatform]],
        action: np.ndarray,
        rl_inputs: dict[str, Any],
    ):
        """
        This function assigns TaskModelPlatform with it's chosen_execution based on the offloading_decision.
        """
        for i, task_model_platform in enumerate(task_batch):
            if not task_model_platform or i >= len(action):
                continue

            task_type = task_model_platform.task.type
            combination_idx = action[i]
            # print(f"[INFO] Task {task_model_platform.task_id} of type {task_type} assigned action {combination_idx}")
            # print(f"[DEBUG] Available executions in the vehicle {[item.__dict__ for item in task_model_platform.executions_dict['V'][1].values()]}")
            # print(f"edge {[item.__dict__ for item in task_model_platform.executions_dict['E'][1].values()]}")
            # print(f"cloud {[item.__dict__ for item in task_model_platform.executions_dict['C'][1].values()]}")
            # Determine the chosen execution based on the action index
            if 0 <= combination_idx < 2:
                vehicle_combinations = list(task_model_platform.executions_dict["V"][1].values())
                chosen_execution = vehicle_combinations[combination_idx]
            elif combination_idx < 8:
                edge_combinations = list(task_model_platform.executions_dict["E"][1].values())
                chosen_execution = edge_combinations[combination_idx - 2]
            elif combination_idx <= 19:
                cloud_combinations = list(task_model_platform.executions_dict["C"][1].values())
                chosen_execution = cloud_combinations[combination_idx - 8]
            else:
                print(
                    f"[WARNING] Invalid action {combination_idx} for task {task_model_platform.task_id}"
                )
                continue

            task_model_platform.chosen_execution = chosen_execution
    
    def _fallback_decisions(self, task_batch: list[Optional[TaskModelPlatform]]) -> list[Optional[TaskModelPlatform]]:
        """
        Assign random offloading decisions for tasks when RL model is unavailable.
        Randomly selects a valid execution (Vehicle, Edge, or Cloud) for each task.
        """
        for task_model_platform in task_batch:
            if not task_model_platform:
                continue

            # Get all available executions for the task
            vehicle_executions = list(task_model_platform.executions_dict.get("V", ({"is_calculated": False}, {}))[1].values())
            edge_executions = list(task_model_platform.executions_dict.get("E", ({"is_calculated": False}, {}))[1].values())
            cloud_executions = list(task_model_platform.executions_dict.get("C", ({"is_calculated": False}, {}))[1].values())

            # Build a combined list and record index ranges for each group
            all_executions = []
            ranges = {}

            if vehicle_executions:
                start = len(all_executions)
                all_executions.extend(vehicle_executions)
                ranges["vehicle"] = range(start, len(all_executions))

            if cloud_executions:
                start = len(all_executions)
                all_executions.extend(cloud_executions)
                ranges["cloud"] = range(start, len(all_executions))

            if edge_executions:
                start = len(all_executions)
                all_executions.extend(edge_executions)
                ranges["edge"] = range(start, len(all_executions))

            chosen_execution = None
            if all_executions:
                # Base probabilities
                base_probs = {"vehicle": 0.6, "cloud": 0.2, "edge": 0.3}

                # Filter only present groups and normalize weights
                available_groups = list(ranges.keys())
                weights = [base_probs[g] for g in available_groups]
                total = sum(weights)
                weights = [w / total for w in weights]

                # Choose group then random index from its range
                chosen_group = random.choices(available_groups, weights=weights, k=1)[0]
                chosen_index = random.choice(list(ranges[chosen_group]))
                chosen_execution = all_executions[chosen_index]

                task_model_platform.chosen_execution = chosen_execution
                print(
                    f"[INFO] Fallback: Task {task_model_platform.task_id} assigned to "
                    f"{chosen_execution.level} on {chosen_execution.platform.name}"
                )
            else:
                print(
                    f"[WARNING] Fallback: No valid executions for task {task_model_platform.task_id}"
                )

        return task_batch


class PPOModel(OffloaingAlgorithme):
    def __init__(self):
        # Initialize Ray with compatibility settings
        ray.init(ignore_reinit_error=True)

        # Register custom components
        register_custom_components()

        self.config = config = {
            "env": "OffloadingAdEnv-v0",
            "framework": "tf2",
            "num_gpus": 0,  # Start with CPU, switch to GPU if available
            "num_workers": 0,
            "model": {
                "custom_model": "CustomTFModel",
                "custom_model_config": {"config": OffloadingAdEnv().config},
            },
            "logger_creator": lambda config: NoopLogger(
                config, "/tmp/ray_results"
            ),  # Replace with your logdir if needed
        }
        self.checkpoint_path = "/Users/kyorakuna/Desktop/New folder/ad_sim/backend/final_model"  # Update with actual path
        self.model = self._load_rl_model()

    def _load_rl_model(self):
        """Load model with proper serialization handling."""
        # Ensure TensorFlow uses legacy Keras if needed
        os.environ["TF_USE_LEGACY_KERAS"] = "1"

        # Fix absolute remote state_file paths from imported checkpoint metadata.
        json_path = os.path.join(self.checkpoint_path, "rllib_checkpoint.json")
        local_state_file = os.path.join(self.checkpoint_path, "algorithm_state.pkl")
        if os.path.exists(json_path) and os.path.exists(local_state_file):
            try:
                with open(json_path, "r") as f:
                    checkpoint_meta = json.load(f)
                if (
                    isinstance(checkpoint_meta, dict)
                    and checkpoint_meta.get("type") == "Algorithm"
                    and checkpoint_meta.get("state_file") != local_state_file
                ):
                    checkpoint_meta["state_file"] = local_state_file
                    with open(json_path, "w") as f:
                        json.dump(checkpoint_meta, f, indent=2)
                    print(
                        f"[INFO] Rewrote checkpoint state_file in {json_path} to local path."
                    )
            except Exception as exc:
                print(
                    f"[WARNING] Could not normalize checkpoint metadata {json_path}: {exc}"
                )

        algo = PPO(config=self.config)  # type: ignore

        try:
            algo.restore(self.checkpoint_path)
        except Exception as exc:
            print(
                f"[WARNING] RL checkpoint restore failed: {exc}. "
                "Falling back to random offloading decisions."
            )
            return None
        return algo

    def calculate_offloading_decisions(
        self, task_batch: list[Optional[TaskModelPlatform]], rl_inputs: dict
    ) -> list[Optional[TaskModelPlatform]]:
        """
        Use the trained RL model to make offloading decisions.
        """
        if not self.model:
            print("[ERROR] RL model not loaded. Using random decisions.")
            return self._fallback_decisions(task_batch)

        try:
            # print("++++++++++++++++++++++++++++++++++")
            # print(len(rl_inputs))
            # Get action from RL model
            start_time = time.time()
            action = self.model.compute_single_action(
                observation=rl_inputs,
                explore=True,  # Use exploitation for inference
                policy_id="default_policy",
            )
            inference_time = time.time() - start_time

            if action is None:
                return self._fallback_decisions(task_batch)

            print(f"[INFO] RL inference completed in {inference_time:.4f}s")

            # Process the action to assign chosen executions
            self._assign_chosen_executions(task_batch, action, rl_inputs)  # type: ignore

            return task_batch

        except Exception as e:
            print(f"[ERROR] RL inference failed: {e}. Using fallback decisions.")
            return self._fallback_decisions(task_batch)
