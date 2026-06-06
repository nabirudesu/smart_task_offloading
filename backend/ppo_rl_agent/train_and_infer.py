import tensorflow as tf
import numpy as np
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict
from gym.spaces import Dict, Tuple, Discrete, Box, MultiDiscrete, MultiBinary
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.models import ModelCatalog
import gymnasium as gym
import os
import shutil
from typing import Any
from utils.config import config, norm_params
from utils.data_cost_model import combinations_cost_model
import utils.data_generator as data_gen
from utils.offloading_ad_env import OffloadingAdEnv

# Configurable parameters
N_MAX = 10  # Maximum number of tasks
V_MAX = 5  # Maximum number of vehicles


class CustomPPOModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomPPOModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.task_types = config.get("task_types")
        self.hardware_names = config.get("hardware_names")
        self.models_names = config.get("MODEL_NAMES")
        self.levels = config.get("LEVELS")
        self.max_combinations_per_type = config.get("Max_combinations_per_type")
        self.max_combinations_per_edge = config.get("Max_combinations_per_edge")
        self.n_max = config.get("N_max")
        self.v_max = config.get("V_max")
        self.combinations = combinations_cost_model

        # Task MLP
        self.task_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
            ]
        )

        # Vehicle MLP
        self.vehicle_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )

        # Combination MLP
        self.combination_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )

        # Edge MLP
        self.edge_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )

        # Cloud MLP
        self.cloud_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )

        # Network MLP
        self.network_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )

        # Task Encoder (MultiHeadAttention)
        self.task_encoder = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=128)

        # Task-Combination Binder MLP
        self.binder_mlp = tf.keras.Sequential(
            [tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dense(1)]
        )

        # Fusion MLP
        self.fusion_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
            ]
        )

        # Action Head
        self.action_head = tf.keras.layers.Dense(self.max_combinations_per_type)

        # Value Head
        self.value_head = tf.keras.Sequential(
            [tf.keras.layers.Dense(512, activation="relu"), tf.keras.layers.Dense(1)]
        )

        # Location indices for each combination
        self.loc_indices = {}
        for task_type in self.task_types:
            loc_indices = []
            for comb in self.combinations.get(task_type, [])[: self.max_combinations_per_type]:
                loc_indices.append(comb["niveau_id"] + 1)  # 1=Vehicle, 2=Edge, 3=Cloud
            while len(loc_indices) < self.max_combinations_per_type:
                loc_indices.append(0)  # Padding
            self.loc_indices[task_type] = tf.constant(loc_indices, dtype=tf.int32)

    def flatten_observation(self, obs: dict[str, Any]) -> tuple[tf.Tensor, ...]:
        """Flatten nested observation into tensors."""
        # Tasks: [b, N_max, 13]
        tasks = []
        for i in range(0, self.n_max):
            task = obs["Tasks_Data"][i]
            task_features = [
                tf.cast(task["id_Tache"], tf.float32),
                tf.cast(task["Type"], tf.float32),
                task["min_accuracy"][0],
                task["time_to_deadline"][0],
                task["data_size_input"][0],
                task["data_size_output"][0],
                *task["alpha_list_edge_comb"],
                tf.cast(task["vehicle_idx"], tf.float32),
            ]
            tasks.append(tf.stack(task_features))
        tasks_tensor = tf.stack(tasks, axis=1)  # [b, N_max, 13]

        # Vehicles: [b, V_max, 8]
        vehicles = []
        for i in range(self.v_max):
            vehicle = obs["Vehicles_Data"][i]
            vehicle_features = [tf.cast(vehicle["id_Vehicule"], tf.float32)]
            for hw in vehicle["Hardware_embarques"]:
                vehicle_features.extend(
                    [
                        tf.cast(hw["hardware_name_id"], tf.float32),
                        hw["charge_remaining_percentage"][0],
                        hw["memory_capacity_remaining_of_this_hardware"][0],
                    ]
                )
            while len(vehicle_features) < 8:  # Pad if fewer hardware
                vehicle_features.extend([0.0, 0.0, 0.0])
            vehicles.append(tf.stack(vehicle_features[:8]))
        vehicles_tensor = tf.stack(vehicles, axis=1)  # [b, V_max, 8]

        # Combinations: [b, 5, 20, 7]
        combinations = []
        for task_type in self.task_types:
            type_combs = []
            for comb in obs["Combinations_Data"][task_type][: self.max_combinations_per_type]:
                comb_features = [
                    tf.cast(comb["model_name_id"], tf.float32),
                    tf.cast(comb["hardware_name_id"], tf.float32),
                    comb["accuracy"][0],
                    comb["execution_time"][0],
                    comb["power_consumption"][0],
                    comb["memory_consumption"][0],
                    comb["utilization_percentage"][0],
                ]
                type_combs.append(tf.stack(comb_features))
            while len(type_combs) < self.max_combinations_per_type:
                type_combs.append(tf.zeros(7, dtype=tf.float32))
            combinations.append(tf.stack(type_combs))
        combinations_tensor = tf.stack(combinations, axis=1)  # [b, 5, 20, 7]

        # Edge: [b, 12]
        edge_features = [
            obs["Info_Edge"]["power_P_e_e"][0],
            obs["Info_Edge"]["power_P_e_v"][0],
            obs["Info_Edge"]["power_P_e_c"][0],
        ]
        for hw in obs["Info_Edge"]["Hardware_Edge"]:
            edge_features.extend(
                [
                    tf.cast(hw["hardware_name_id"], tf.float32),
                    hw["charge_remaining_percentage"][0],
                    hw["memory_capacity_remaining_of_this_hardware"][0],
                ]
            )
        while len(edge_features) < 12:  # Pad if fewer hardware
            edge_features.extend([0.0, 0.0, 0.0])
        edge_tensor = tf.stack(edge_features[:12])  # [b, 12]

        # Cloud: [b, 13]
        cloud_features = [obs["Info_Cloud"]["power_P_c_e"][0]]
        for hw in obs["Info_Cloud"]["Hardware_Cloud"]:
            cloud_features.extend(
                [
                    tf.cast(hw["hardware_name_id"], tf.float32),
                    hw["charge_remaining_percentage"][0],
                    hw["memory_capacity_remaining_of_this_hardware"][0],
                ]
            )
        while len(cloud_features) < 13:  # Pad if fewer hardware
            cloud_features.extend([0.0, 0.0, 0.0])
        cloud_tensor = tf.stack(cloud_features[:13])  # [b, 13]

        # Communication: [b, 5]
        communication = [
            obs["Info_Communication"]["vehicle_to_edge_throughput"][0],
            obs["Info_Communication"]["edge_to_edge_throughput"][0],
            obs["Info_Communication"]["edge_to_cloud_throughput"][0],
            obs["Info_Communication"]["edge_to_vehicle_throughput"][0],
            obs["Info_Communication"]["cloud_to_edge_throughput"][0],
        ]
        communication_tensor = tf.stack(communication)  # [b, 5]

        return (
            tasks_tensor,
            vehicles_tensor,
            combinations_tensor,
            edge_tensor,
            cloud_tensor,
            communication_tensor,
        )

    def forward(self, input_dict, state, seq_lens):
        # Flatten observation
        print(self.__dic)
        tasks, vehicles, combinations, edge, cloud, communication = self.flatten_observation(
            input_dict["obs"]
        )

        # Task MLP: [b, N_max, 13] -> [b, N_max, 128]
        task_emb = self.task_mlp(tasks)  # [b, N_max, 128]

        # Vehicle MLP: [b, V_max, 8] -> [b, V_max, 64]
        vehicle_emb = self.vehicle_mlp(vehicles)  # [b, V_max, 64]

        # Combination MLP: [b, 5, 20, 7] -> [b, 5, 20, 64]
        combination_emb = self.combination_mlp(combinations)  # [b, 5, 20, 64]

        # Edge MLP: [b, 12] -> [b, 64]
        edge_emb = self.edge_mlp(edge)  # [b, 64]

        # Cloud MLP: [b, 13] -> [b, 64]
        cloud_emb = self.cloud_mlp(cloud)  # [b, 64]

        # Network MLP: [b, 5] -> [b, 64]
        network_emb = self.network_mlp(communication)  # [b, 64]

        # Task Encoder
        task_emb_updated = self.task_encoder(task_emb, task_emb, task_emb)  # [b, N_max, 128]

        # Task-Combination Binder
        task_types = tf.cast(tasks[:, :, 1], tf.int32)  # [b, N_max]
        valid_type_mask = tf.logical_and(task_types >= 0, task_types < len(self.task_types))
        task_types = tf.where(valid_type_mask, task_types, tf.zeros_like(task_types))

        # Gather combinations and location indices
        selected_comb_emb = tf.gather(
            combination_emb, task_types, batch_dims=1, axis=1
        )  # [b, N_max, 20, 64]
        selected_loc_indices = []
        for batch_idx in range(tf.shape(task_types)[0]):
            batch_loc_indices = []
            for task_idx in range(self.n_max):
                task_type_idx = task_types[batch_idx, task_idx].numpy()
                task_type_str = (
                    self.task_types[task_type_idx]
                    if task_type_idx < len(self.task_types)
                    else self.task_types[0]
                )
                batch_loc_indices.append(self.loc_indices[task_type_str])
            selected_loc_indices.append(tf.stack(batch_loc_indices))
        selected_loc_indices = tf.stack(selected_loc_indices)  # [b, N_max, 20]

        # Binder computation
        task_emb_expanded = tf.expand_dims(task_emb_updated, axis=2)  # [b, N_max, 1, 128]
        task_emb_expanded = tf.tile(
            task_emb_expanded, [1, 1, self.max_combinations_per_type, 1]
        )  # [b, N_max, 20, 128]
        binder_input = tf.concat(
            [task_emb_expanded, selected_comb_emb], axis=-1
        )  # [b, N_max, 20, 192]
        binder_scores = self.binder_mlp(binder_input)  # [b, N_max, 20, 1]
        binder_scores = tf.squeeze(binder_scores, axis=-1)  # [b, N_max, 20]
        attention_weights = tf.nn.softmax(binder_scores, axis=-1)  # [b, N_max, 20]
        bound_features = tf.reduce_sum(
            selected_comb_emb * tf.expand_dims(attention_weights, axis=-1), axis=2
        )  # [b, N_max, 64]

        # Vehicle Indexing
        vehicle_idx = tf.cast(tasks[:, :, -1], tf.int32)  # [b, N_max]
        valid_vehicle_mask = tf.logical_and(
            vehicle_idx >= 0, vehicle_idx < self.v_max
        )  # [b, N_max]
        vehicle_idx = tf.where(valid_vehicle_mask, vehicle_idx, tf.zeros_like(vehicle_idx))
        vehicle_features = tf.gather(
            vehicle_emb, vehicle_idx, batch_dims=1, axis=1
        )  # [b, N_max, 64]

        # Per-Combination Location States
        edge_emb_expanded = tf.tile(
            tf.expand_dims(edge_emb, axis=1), [1, self.n_max, 1]
        )  # [b, N_max, 64]
        cloud_emb_expanded = tf.tile(
            tf.expand_dims(cloud_emb, axis=1), [1, self.n_max, 1]
        )  # [b, N_max, 64]
        loc_states = tf.stack(
            [vehicle_features, edge_emb_expanded, cloud_emb_expanded], axis=2
        )  # [b, N_max, 3, 64]
        loc_indices_adjusted = tf.maximum(selected_loc_indices - 1, 0)  # [b, N_max, 20]
        comb_loc_states = tf.gather(
            loc_states, loc_indices_adjusted, batch_dims=2, axis=2
        )  # [b, N_max, 20, 64]

        # Network State Application
        network_mask = tf.cast(selected_loc_indices > 1, tf.float32)  # [b, N_max, 20]
        network_emb_expanded = tf.tile(
            tf.expand_dims(tf.expand_dims(network_emb, axis=1), axis=2),
            [1, self.n_max, self.max_combinations_per_type, 1],
        )  # [b, N_max, 20, 64]
        comb_loc_states_updated = (
            comb_loc_states + network_mask[:, :, :, tf.newaxis] * network_emb_expanded
        )  # [b, N_max, 20, 64]

        # Feature Concatenation
        loc_states_agg = tf.reduce_mean(comb_loc_states_updated, axis=2)  # [b, N_max, 64]
        concat_features = tf.concat(
            [task_emb_updated, bound_features, vehicle_features, loc_states_agg], axis=-1
        )  # [b, N_max, 320]

        # Fusion MLP
        fused = self.fusion_mlp(concat_features)  # [b, N_max, 128]

        # Action Head
        logits = self.action_head(fused)  # [b, N_max, 20]
        valid_mask = tf.expand_dims(
            tf.cast(valid_vehicle_mask, tf.float32), axis=-1
        )  # [b, N_max, 1]
        logits = logits * valid_mask + tf.where(valid_mask, 0.0, -1e9)  # Mask padded tasks

        # Value Head
        value_input = tf.concat(
            [
                tf.reshape(fused, [-1, self.n_max * 128]),
                tf.reshape(vehicle_emb, [-1, self.v_max * 64]),
            ],
            axis=-1,
        )
        value = self.value_head(value_input)  # [b, 1]

        self._value_out = value
        return logits, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


# Training script
def train_ppo():
    # Clean up Ray temporary directory to avoid conflicts
    ray_temp_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", "ray")
    if os.path.exists(ray_temp_dir):
        try:
            shutil.rmtree(ray_temp_dir)
            print(f"Cleared Ray temporary directory: {ray_temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clear Ray temp directory: {e}")

    # Initialize Ray with dashboard disabled
    ray.shutdown()
    try:
        ray.init(ignore_reinit_error=True, dashboard_host="")
        print("Ray initialized successfully.")
    except Exception as e:
        print(f"Error initializing Ray: {e}")
        return

    ModelCatalog.register_custom_model("custom_ppo_model", CustomPPOModel)

    config_ppo = {
        "env": OffloadingAdEnv,
        "framework": "tf2",
        "model": {
            "custom_model": "custom_ppo_model",
            "custom_model_config": {
                "N_max": N_MAX,
                "V_max": V_MAX,
            },
        },
        "api_stack": {
            "enable_rl_module_and_learner": False,
            "enable_env_runner_and_connector_v2": False,
        },
        "num_workers": 2,
        "num_gpus": 0,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 30,
        "rollout_fragment_length": 200,
        "lr": 5e-5,
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "vf_clip_param": 10.0,
        "num_envs_per_env_runner": 1,
        "batch_mode": "truncate_episodes",
        "experimental": {"_validate_config": False},
        # Add configurations to disable the new API stack
        "_enable_rl_module_api": False,
        "_enable_learner_api": False,
    }

    try:
        config_ppo = PPOConfig().update_from_dict(config_ppo)
        trainer = config_ppo.build()
        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        for i in range(100):  # Number of training iterations
            result = trainer.train()
            print(f"Iteration {i}:")
            print(f"  Mean episode reward: {result['episode_reward_mean']}")
            print(f"  Episode length: {result['episode_len_mean']}")
            if i % 10 == 0:
                checkpoint = trainer.save(checkpoint_dir)
                print(f"Checkpoint saved at {checkpoint}")
    except Exception as e:
        print(f"Training failed: {e}")
    finally:
        ray.shutdown()


# Inference script
def run_inference(checkpoint_path):
    # Clean up Ray temporary directory
    ray_temp_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", "ray")
    if os.path.exists(ray_temp_dir):
        try:
            print(f"Cleared Ray temporary directory: {ray_temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clear Ray temp directory: {e}")

    ray.shutdown()
    # Initialize Ray with dashboard disabled
    try:
        ray.init(ignore_reinit_error=True, dashboard_host="")
        print("Ray initialized successfully.")
    except Exception as e:
        print(f"Error initializing Ray: {e}")
        return

    ModelCatalog.register_custom_model("custom_ppo_model", CustomPPOModel)

    config_ppo = {
        "env": OffloadingAdEnv,
        "framework": "tf2",
        "model": {"custom_model": "custom_ppo_model", "custom_model_config": {}},
        "api_stack": {
            "enable_rl_module_and_learner": False,
            "enable_env_runner_and_connector_v2": False,
        },
        "num_workers": 0,
        # Add configurations to disable the new API stack for inference as well
        "_enable_rl_module_api": False,
        "_enable_learner_api": False,
    }

    try:
        trainer = PPO(config=config_ppo)
        trainer.restore(checkpoint_path)

        env = OffloadingAdEnv()
        obs = env.reset()[0]
        done = False
        total_reward = 0.0

        while not done:
            action = trainer.compute_single_action(obs, explore=False)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()

        print(f"Total reward: {total_reward}")
        env.close()
    except Exception as e:
        print(f"Inference failed: {e}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    # Run training
    train_ppo()

    # Run inference (example with latest checkpoint)
    # latest_checkpoint = "./checkpoints/checkpoint_000100/checkpoint-100"  # Update with actual path
    # run_inference(latest_checkpoint)
