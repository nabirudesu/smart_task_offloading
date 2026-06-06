import json
import logging
import os

import numpy as np
import tensorflow as tf
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from gymnasium import spaces
from typing import Dict, Any
from ray.rllib.utils import override
from ray import tune
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ppo_rl_agent.flat_env import OffloadingAdEnv
from ppo_rl_agent.config import config, norm_params

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# Dynamically set CUDA_VISIBLE_DEVICES based on GPU availability
if tf.config.list_physical_devices("GPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logger.info("GPU detected. Setting CUDA_VISIBLE_DEVICES=0")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    logger.info("No GPU detected. Disabling CUDA_VISIBLE_DEVICES")
os.environ["TF_USE_LEGACY_KERAS"] = "1"
tf1, tf, tfv = try_import_tf()


# Ray actor for shared metrics
@ray.remote
class MetricsActor:
    def __init__(self):
        self.episode_metrics = []

    def append_metrics(self, metrics):
        self.episode_metrics.append(metrics)
        return len(self.episode_metrics)

    def get_metrics(self):
        return self.episode_metrics

    def clear_metrics(self):
        metrics = self.episode_metrics
        self.episode_metrics = []
        return metrics


# Custom callback for metrics collection
class CustomMetricsCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.metrics_actor = ray.get_actor("metrics_actor")
        self.callback_id = id(self)

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        try:
            # Get custom metrics from the episode
            custom_metrics = episode._last_infos.get("agent0", {})

            # Create metrics dictionary
            metrics = {
                "episode_id": episode.episode_id,
                "episode_length": episode.length,
                "episode_reward": episode.total_reward,
                **custom_metrics,
            }

            # Add to actor
            ray.get(self.metrics_actor.append_metrics.remote(metrics))

        except Exception as e:
            logger.error(f"Error in on_episode_end: {e}")
        # print(f"Custom Metrics:{metrics}")
        # if hasattr(worker, 'algorithm') and hasattr(worker.algorithm, 'metrics_actor'):
        #     size = ray.get(worker.algorithm.metrics_actor.append_metrics.remote(metrics))
        #     logger.info(f"(Callback) Episode {episode.episode_id} ended with metrics: {metrics}")
        #     logger.info(f"(Callback) Callback ID: {self.callback_id}")
        #     logger.info(f"(Callback) Actor metrics size: {size}")

    def on_train_result(self, *, algorithm, result, **kwargs):
        if hasattr(algorithm, "metrics_actor"):
            metrics = ray.get(algorithm.metrics_actor.get_metrics.remote())
            result["custom_episode_metrics"] = metrics


class CustomTFModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, config):
        super(CustomTFModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.config = config
        self.task_types = config["task_types"]
        self.hardware_names = config["hardware_names"]
        self.models_names = config["MODEL_NAMES"]
        self.levels = config["LEVELS"]
        self.N_max = config["N_max"]
        self.V_max = config["V_max"]
        self.max_combinations_per_type = config["Max_combinations_per_type"]
        self.max_combinations_per_edge = config["Max_combinations_per_edge"]
        self.num_hardware_vehicle = len(config["Hardware_Vehicle"])
        self.num_hardware_edge = len(config["Hardware_Edge"])
        self.num_hardware_cloud = len(config["Hardware_Cloud"])

        # Build MLPs
        self.task_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
            ]
        )
        self.vehicle_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )
        self.combination_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )
        self.edge_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )
        self.cloud_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )
        self.network_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )
        self.task_encoder = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=128)
        self.binder_mlp = tf.keras.Sequential(
            [tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dense(1)]
        )
        self.fusion_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(20, activation="relu"),
            ]
        )
        self.action_net = tf.keras.layers.Dense(self.max_combinations_per_type, activation=None)
        self.value_net = tf.keras.Sequential(
            [tf.keras.layers.Dense(512, activation="relu"), tf.keras.layers.Dense(1)]
        )
        dummy_bsz = 1
        self.task_mlp(tf.zeros((dummy_bsz, self.N_max, 13)))
        self.vehicle_mlp(tf.zeros((dummy_bsz, self.V_max, 8)))
        self.combination_mlp(
            tf.zeros((dummy_bsz, len(self.task_types), self.max_combinations_per_type, 7))
        )
        self.edge_mlp(tf.zeros((dummy_bsz, 18)))
        self.cloud_mlp(tf.zeros((dummy_bsz, 16)))
        self.network_mlp(tf.zeros((dummy_bsz, 5)))
        self.binder_mlp(tf.zeros((dummy_bsz, 192)))
        self.task_encoder(
            tf.zeros((dummy_bsz, self.N_max, 128)), tf.zeros((dummy_bsz, self.N_max, 128))
        )
        self.fusion_mlp(tf.zeros((dummy_bsz, self.N_max, 320)))
        self.action_net(tf.zeros((dummy_bsz, self.N_max, self.max_combinations_per_type)))
        self.value_net(tf.zeros((dummy_bsz, self.N_max * 20 + self.V_max * 64)))

    @override(TFModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        tasks_data = obs["Tasks_Data"]
        vehicles_data = obs["Vehicles_Data"]
        combinations_data = obs["Combinations_Data"]
        info_edge = obs["Info_Edge"]
        info_cloud = obs["Info_Cloud"]
        info_communication = obs["Info_Communication"]

        task_input = tf.concat(
            [
                tf.cast(tasks_data["id_Tache"], tf.float32),
                tf.cast(tasks_data["Type"], tf.float32),
                tasks_data["min_accuracy"],
                tasks_data["time_to_deadline"],
                tasks_data["data_size_input"],
                tasks_data["data_size_output"],
                tf.cast(tasks_data["alpha_list_edge_comb"], tf.float32),
                tf.cast(tasks_data["vehicle_idx"], tf.float32),
            ],
            axis=-1,
        )
        task_emb = self.task_mlp(task_input)

        vehicle_input = tf.concat(
            [
                tf.tile(
                    tf.expand_dims(vehicles_data["power_v_e"], axis=2),
                    [1, 1, self.num_hardware_vehicle, 1],
                ),
                tf.cast(vehicles_data["hardware_name_id"], tf.float32),
                vehicles_data["charge_remaining_percentage"],
                vehicles_data["memory_capacity_remaining_of_this_hardware"],
            ],
            axis=-1,
        )
        vehicle_input = tf.reshape(vehicle_input, [-1, self.V_max, self.num_hardware_vehicle * 4])
        vehicle_emb = self.vehicle_mlp(vehicle_input)

        comb_features = []
        for task_type in self.task_types:
            comb = combinations_data[task_type]
            comb_input = tf.concat(
                [
                    tf.cast(comb["model_name_id"], tf.float32),
                    tf.cast(comb["hardware_name_id"], tf.float32),
                    comb["accuracy"],
                    comb["execution_time"],
                    comb["power_consumption"],
                    comb["memory_consumption"],
                    comb["utilization_percentage"],
                ],
                axis=-1,
            )
            comb_features.append(comb_input)
        comb_input = tf.stack(comb_features, axis=1)
        comb_emb = self.combination_mlp(comb_input)

        edge_input = tf.concat(
            [
                tf.tile(
                    tf.expand_dims(info_edge["power_P_e_e"], axis=2), [1, self.num_hardware_edge, 1]
                ),
                tf.tile(
                    tf.expand_dims(info_edge["power_P_e_v"], axis=2), [1, self.num_hardware_edge, 1]
                ),
                tf.tile(
                    tf.expand_dims(info_edge["power_P_e_c"], axis=2), [1, self.num_hardware_edge, 1]
                ),
                tf.cast(info_edge["hardware_name_id"], tf.float32),
                info_edge["charge_remaining_percentage"],
                info_edge["memory_capacity_remaining_of_this_hardware"],
            ],
            axis=-1,
        )
        edge_input = tf.reshape(edge_input, [edge_input.shape[0], -1])
        edge_emb = self.edge_mlp(edge_input)

        cloud_input = tf.concat(
            [
                tf.tile(
                    tf.expand_dims(info_cloud["power_P_c_e"], axis=2),
                    [1, self.num_hardware_cloud, 1],
                ),
                tf.cast(info_cloud["hardware_name_id"], tf.float32),
                info_cloud["charge_remaining_percentage"],
                info_cloud["memory_capacity_remaining_of_this_hardware"],
            ],
            axis=-1,
        )
        cloud_input = tf.reshape(cloud_input, [tf.shape(cloud_input)[0], -1])
        cloud_emb = self.cloud_mlp(cloud_input)

        network_input = tf.concat(
            [
                info_communication["vehicle_to_edge_throughput"],
                info_communication["edge_to_edge_throughput"],
                info_communication["edge_to_cloud_throughput"],
                info_communication["edge_to_vehicle_throughput"],
                info_communication["cloud_to_edge_throughput"],
            ],
            axis=-1,
        )
        network_emb = self.network_mlp(network_input)

        task_emb_updated = self.task_encoder(task_emb, task_emb)
        task_types = tf.cast(tasks_data["Type"], tf.int32)
        task_types = tf.squeeze(task_types, axis=-1)
        valid_type_mask = tf.logical_and(task_types >= 0, task_types < len(self.task_types))
        task_types = tf.where(valid_type_mask, task_types, 0)
        loc_indices = tf.stack(
            [combinations_data[task_type]["niveau_id"] for task_type in self.task_types], axis=1
        )
        loc_indices = tf.squeeze(loc_indices, axis=-1)
        selected_comb_emb = tf.gather(comb_emb, task_types, axis=1, batch_dims=1)
        selected_loc_indices = tf.gather(loc_indices, task_types, axis=1, batch_dims=1)
        task_emb_expanded = tf.expand_dims(task_emb_updated, axis=2)
        task_emb_expanded = tf.tile(task_emb_expanded, [1, 1, self.max_combinations_per_type, 1])
        binder_input = tf.concat([task_emb_expanded, selected_comb_emb], axis=-1)
        scores = self.binder_mlp(binder_input)
        attention_weights = tf.nn.softmax(scores, axis=2)
        bound_features = tf.reduce_sum(selected_comb_emb * attention_weights, axis=2)

        vehicle_idx = tf.cast(tasks_data["vehicle_idx"], tf.int32)
        vehicle_idx = tf.squeeze(vehicle_idx, axis=-1)
        valid_vehicle_mask = tf.logical_and(vehicle_idx >= 0, vehicle_idx < self.V_max)
        vehicle_idx = tf.where(valid_vehicle_mask, vehicle_idx, 0)
        vehicle_features = tf.gather(vehicle_emb, vehicle_idx, axis=1, batch_dims=1)

        edge_emb_expanded = tf.expand_dims(edge_emb, axis=1)
        edge_emb_expanded = tf.tile(edge_emb_expanded, [1, self.N_max, 1])
        cloud_emb_expanded = tf.expand_dims(cloud_emb, axis=1)
        cloud_emb_expanded = tf.tile(cloud_emb_expanded, [1, self.N_max, 1])
        loc_states = tf.stack([vehicle_features, edge_emb_expanded, cloud_emb_expanded], axis=2)
        selected_loc_indices_adj = tf.cast(selected_loc_indices, tf.int32)
        comb_loc_states = tf.gather(loc_states, selected_loc_indices_adj, axis=2, batch_dims=2)

        network_mask = tf.cast(selected_loc_indices > 1, tf.float32)[..., tf.newaxis]
        network_emb_expanded = tf.expand_dims(network_emb, axis=1)
        network_emb_expanded = tf.expand_dims(network_emb_expanded, axis=2)
        network_emb_expanded = tf.tile(
            network_emb_expanded, [1, self.N_max, self.max_combinations_per_type, 1]
        )
        comb_loc_states_updated = comb_loc_states + network_mask * network_emb_expanded

        loc_states_agg = tf.reduce_mean(comb_loc_states_updated, axis=2)
        concat_features = tf.concat(
            [task_emb_updated, bound_features, vehicle_features, loc_states_agg], axis=-1
        )

        fused = self.fusion_mlp(concat_features)
        logits = self.action_net(fused)
        valid_mask_bool = tf.expand_dims(valid_vehicle_mask, axis=-1)
        valid_mask = tf.cast(valid_mask_bool, tf.float32)
        logits = logits * valid_mask + tf.where(valid_mask_bool, 0.0, -1e9)
        logits_flat = tf.reshape(logits, [-1, self.N_max * self.max_combinations_per_type])

        features_flat = tf.reshape(fused, [-1, self.N_max * 20])
        vehicle_emb_flat = tf.reshape(vehicle_emb, [-1, self.V_max * 64])
        value_input = tf.concat([features_flat, vehicle_emb_flat], axis=-1)
        self._value_out = self.value_net(value_input)

        return logits_flat, state

    @override(TFModelV2)
    def value_function(self):
        return tf.squeeze(self._value_out, axis=-1)


# Training script
if __name__ == "__main__":
    checkpoint_dir = "/content/drive/Shareddrives/offloading_system"
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_checkpoint_path = os.path.join(checkpoint_dir, "_final_model")
    metrics_file_path = os.path.join(checkpoint_dir, "training_metrics.json")

    # Explicitly register model
    ModelCatalog.register_custom_model("CustomTFModel", CustomTFModel)
    # Initialize Ray without address="auto" to start a fresh cluster
    ray.init(
        ignore_reinit_error=True,
        num_cpus=8,
        num_gpus=1 if tf.config.list_physical_devices("GPU") else 0,
    )
    logger.info(f"Ray initialized. Cluster resources: {ray.cluster_resources()}")
    logger.info(f"Ray worker info: {ray.get_runtime_context().worker.__dict__}")

    try:
        ray.kill(ray.get_actor("metrics_actor"))
    except ValueError:
        pass

    metrics_actor = MetricsActor.options(name="metrics_actor").remote()

    def env_creator(env_config):
        return OffloadingAdEnv()

    register_env("OffloadingAdEnv-v0", env_creator)

    config_ppo = {
        "env": "OffloadingAdEnv-v0",
        "framework": "tf2",
        "num_workers": 1,  # Driver-only to simplify
        "num_gpus": 1 if tf.config.list_physical_devices("GPU") else 0,
        "num_gpus_per_worker": 1,
        "num_envs_per_worker": 1,
        "rollout_fragment_length": 1024,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 512,
        "num_sgd_iter": 3,
        "lr": 0.00003,
        "gamma": 0.99,
        "lambda": 0.95,
        "entropy_coeff": 0.01,
        "kl_coeff": 0.0,
        "clip_param": 0.2,
        "reuse_actors": True,
        "model": {
            "custom_model": "CustomTFModel",
            "custom_model_config": {"config": env_creator({}).config},
        },
        "keep_per_episode_custom_metrics": True,
        "evaluation_interval": 1,
        "evaluation_duration": 10,
        "evaluation_duration_unit": "episodes",
        "evaluation_config": {
            "record_env": True,
            "render_env": False,
        },
        "report_per_episode_metrics": True,
        "callbacks": CustomMetricsCallback,
    }

    training_metrics = []
    if os.path.exists(metrics_file_path):
        try:
            with open(metrics_file_path, "r") as f:
                training_metrics = json.load(f)
            logger.info(f"Loaded existing metrics from {metrics_file_path}")
        except Exception as e:
            logger.error(f"Error loading metrics file: {e}. Starting with empty metrics.")

    algo = PPO(config=config_ppo)
    print(algo.config.__dict__)
    algo.metrics_actor = metrics_actor

    if os.path.exists(final_checkpoint_path):
        logger.info(f"Attempting to resume training from: {final_checkpoint_path}")
        # Create new metrics actor
        algo.metrics_actor = metrics_actor
        algo.restore(final_checkpoint_path)

        logger.info(f"Successfully restored checkpoint: {final_checkpoint_path}")
    else:
        logger.info("No checkpoint found. Starting fresh training.")

    for i in range(0, 30):
        result = algo.train()
        current_metrics = ray.get(algo.metrics_actor.get_metrics.remote())
        logger.info(f"Immediate metrics check: {len(current_metrics)} metrics available")

        iteration_metrics = {
            "iteration": i,
            "mean_reward": result["env_runners"]["episode_reward_mean"],
            "min_reward": result["env_runners"]["episode_reward_min"],
            "max_reward": result["env_runners"]["episode_reward_max"],
            "episodes_this_iter": result["env_runners"]["episodes_this_iter"],
        }

        episode_metrics = ray.get(algo.metrics_actor.get_metrics.remote())
        custom_metrics = result["env_runners"].get("custom_metrics", {})
        if not custom_metrics and episode_metrics:
            custom_metrics = {}
            for key in episode_metrics[0].keys():
                if key not in ["episode_id", "episode_length", "episode_reward"]:
                    values = [
                        m[key] for m in episode_metrics if isinstance(m.get(key), (int, float))
                    ]
                    custom_metrics[f"{key}_mean"] = sum(values) / len(values) if values else 0.0
            result["env_runners"]["custom_metrics"] = custom_metrics
            logger.info(f"Aggregated {len(episode_metrics)} episode metrics from actor")

        if custom_metrics:
            iteration_metrics.update(custom_metrics)
        else:
            logger.warning(
                f"No custom metrics in result['env_runners']['custom_metrics'] for iteration {i}"
            )
            logger.debug(f"result['env_runners']: {result['env_runners']}")

        iteration_metrics["episodes"] = episode_metrics
        training_metrics.append(iteration_metrics)

        logger.info(f"Iteration {i}: mean reward = {iteration_metrics['mean_reward']}")
        logger.info(f"Iteration {i}: min reward = {iteration_metrics['min_reward']}")
        logger.info(f"Iteration {i}: max reward = {iteration_metrics['max_reward']}")
        logger.info(f"Iteration {i}: episodes completed = {len(episode_metrics)}")

        ray.get(metrics_actor.clear_metrics.remote())
        ray.get(algo.metrics_actor.clear_metrics.remote())

        if i % 2 == 0:
            # Temporarily remove actor reference
            temp_actor = algo.metrics_actor
            algo.metrics_actor = None

            checkpoint_path = os.path.join(checkpoint_dir, f"_check_point_{i}_")
            checkpoint = algo.save(checkpoint_path)

            algo.metrics_actor = temp_actor
            logger.info(f"Checkpoint saved at {checkpoint}")
        if i % 10 == 0:
            try:
                with open(metrics_file_path, "a") as f:
                    json.dump(training_metrics, f, indent=4)
                logger.info(f"metrics saved to {metrics_file_path} after {i} iterations")
            except Exception as e:
                logger.error(f"Error saving final metrics: {e}")
            algo.save(final_checkpoint_path)
            logger.info(f"Final model saved to: {final_checkpoint_path} after {i} iterations")
            training_metrics = []
    try:
        with open(metrics_file_path, "a") as f:
            json.dump(training_metrics, f, indent=4)
        logger.info(f"Final metrics saved to {metrics_file_path}")
    except Exception as e:
        logger.error(f"Error saving final metrics: {e}")

    algo.save(final_checkpoint_path)
    logger.info(f"Training complete. Final model saved to: {final_checkpoint_path}")
    try:
        ray.kill(ray.get_actor("metrics_actor"))
    except Exception:
        pass

    ray.shutdown()
