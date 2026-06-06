import numpy as np
from typing import Optional
from entities.task_model_platform import TaskModelPlatform, ModelPlatformExecution
import xgboost as xgb
from entities.servers.default_executions import default_executions


class CostModelCache:
    """Cache for cost model predictions with hit rate tracking."""

    def __init__(self):
        self.cache: dict[str, dict[str, float]] = {}
        self.hit_count = 0
        self.miss_count = 0

    def get_estimation(self, key: str) -> dict[str, float]:
        """Retrieve cached prediction if available."""
        if key in self.cache.keys():
            self.hit_count += 1
            return self.cache[key]
        else:
            self.miss_count += 1
            return {}

    def put_estimation(self, key: str, result: dict[str, float]):
        """Store prediction in cache."""
        self.cache[key] = result

    def get_hit_rate(self):
        """Calculate actual cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0


class CostModel:
    def __init__(self, name):
        self.name = name
        self.execution_model = ExecutionModel()
        self.energy_model = EnergyModel()
        self.memory_model = MemoryModel()
        self.usage_model = UsageModel()
        self.cache = CostModelCache()
        self.default_estimations = default_executions
    
    def estimate(self, task_model_platform: TaskModelPlatform,level:str) -> None:
        task_type = task_model_platform.task.type
        for key,executions_list in default_executions.items(): # loop over all types
            if key == task_type:
                for exec_dict in executions_list: # loop over all executions of that type
                    for _, level_execution in task_model_platform.executions_dict[level][1].items():
                        if (level_execution.is_calculated == False 
                            and level_execution.platform.name == exec_dict.get("hardware_name_id") 
                            and level_execution.model.name == exec_dict.get("model_name_id")):
                                level_execution.execution_time = exec_dict.get("execution_time", 0.0)
                                level_execution.memory_consumption = exec_dict.get("memory_consumption", 0.0)
                                level_execution.energy_consumption = exec_dict.get("power_consumption", 0.0)
                                level_execution.platform_usage = exec_dict.get("utilization_percentage", 0.0)
                                level_execution.is_calculated == False
                        # print(f"[INFO] Default estimation applied for {level_execution.__dict__}")

    """
    def estimate(self, task_model_platform: TaskModelPlatform) -> None:
        for (_,level_executions,) in (
            task_model_platform.executions_dict.items()
        ):  # this only creates 3 iterations ("V","E","C")
            if level_executions[0]["is_calculated"] == True:
                continue
            for _, execution in level_executions[1].items():
                cost_estimations = self.cache.get_estimation(execution.id)
                if not cost_estimations:
                    time_features = np.array(
                        [
                            task_model_platform.task.data_size,
                            execution.model.nb_layers,
                            execution.model.sum_activation,
                            execution.model.model_size,
                            execution.platform.ram_size.level,
                            execution.platform.memory_bandwidth,
                            execution.platform.power_efficiency.capacity,
                            execution.platform.peak_flops,
                            execution.platform.processing_units.capacity,
                            execution.platform.num_tensor_cores,
                        ]
                    )
                    memory_features = np.array(
                        [
                            task_model_platform.task.data_size,
                            execution.model.params,
                            execution.model.sum_activation,
                            execution.platform.ram_size.level,
                            execution.platform.memory_bandwidth,
                            execution.platform.power_efficiency.capacity,
                            execution.platform.peak_flops,
                            execution.platform.processing_units.capacity,
                            execution.platform.num_tensor_cores,
                            execution.model.params,
                        ]
                    )
                    power_features = np.array(
                        [
                            task_model_platform.task.data_size,
                            execution.model.params,
                            execution.model.model_flops,
                            execution.model.nb_layers,
                            execution.model.sum_activation,
                            execution.model.weighted_sum_neurons,
                            execution.platform.ram_size.level,
                            execution.platform.memory_bandwidth,
                            execution.platform.power_efficiency.capacity,
                            execution.platform.peak_flops,
                            execution.platform.processing_units.capacity,
                            execution.platform.num_tensor_cores,
                        ]
                    )

                    cost_estimations = {
                        "execution_time": self.execution_model.estimate_time_cost(time_features),
                        "memory_consumption": self.memory_model.estimate_memory_cost(
                            memory_features
                        ),
                        "energy_consumption": self.energy_model.estimate_energy_cost(
                            power_features
                        ),
                    }
                    usage_features = np.array(
                        [
                            execution.model.model_flops,
                            cost_estimations.get("execution_time"),
                            execution.platform.peak_flops,
                        ]
                    )
                    cost_estimations["platform_usage"] = self.usage_model.estimate_usage_cost(
                        usage_features
                    )
                    self.cache.put_estimation(execution.id, cost_estimations)
                else:
                    print(f"[INFO] Cache hit for key: {execution.id}")

                execution.execution_time = cost_estimations.get("execution_time", 0.0)
                execution.memory_consumption = cost_estimations.get("memory_consumption", 0.0)
                execution.energy_consumption = cost_estimations.get("energy_consumption", 0.0)
                execution.platform_usage = cost_estimations.get("platform_usage", 0.0)
            level_executions[0]["is_calculated"] = True
    """


class ExecutionModel:
    def estimate_time_cost(self, features_input: np.ndarray) -> float:
        """Predict execution time using xgboost model."""
        loaded_model = xgb.Booster()
        loaded_model.load_model("/Users/kyorakuna/Desktop/New folder/ad_sim/backend/cost_models/xgboost_model.json")
        feature_names = [
            "input_size",
            "nb_layers",
            "sum_activations",
            "weighted_sum_neurons",
            "gpu_memory",
            "gpu_memory_bandwidth",
            "gpu_power",
            "gpu_flops",
            "gpu_nb_cores",
            "gpu_nb_tensor_cores",
        ]
        dmatrix = xgb.DMatrix(features_input.reshape(1, -1), feature_names=feature_names)
        prediction = loaded_model.predict(dmatrix)
        if prediction.size > 0:
            return float(prediction[0] / 1000)  # Convert to seconds
        else:
            print("[ERROR] Prediction returned empty array.")
            return 0.0


class EnergyModel:
    def estimate_energy_cost(self, features_input: np.ndarray) -> float:
        """Predict execution time using xgboost model."""
        loaded_model = xgb.Booster()
        loaded_model.load_model("/Users/kyorakuna/Desktop/New folder/ad_sim/backend/cost_models/power_xgboost_model.json")
        feature_names = [
            "input_size",
            "nb_params_conv",
            "flops",
            "nb_layers",
            "sum_activations",
            "weighted_sum_neurons",
            "gpu_memory",
            "gpu_memory_bandwidth",
            "gpu_power",
            "gpu_flops",
            "gpu_nb_cores",
            "gpu_nb_tensor_cores",
        ]
        dmatrix = xgb.DMatrix(features_input.reshape(1, -1), feature_names=feature_names)
        prediction = loaded_model.predict(dmatrix)
        if prediction.size > 0:
            return float(prediction[0] / 1000)  # Convert to kJ
        else:
            print("[ERROR] Prediction returned empty array.")
            return 0.0


class MemoryModel:
    def estimate_memory_cost(self, features_input: np.ndarray) -> float:
        """Predict execution time using xgboost model."""
        loaded_model = xgb.Booster()
        loaded_model.load_model("/Users/kyorakuna/Desktop/New folder/ad_sim/backend/cost_models/memory_xgboost_model.json")
        feature_names = [
            "input_size",
            "nb_layers",
            "sum_activations",
            "gpu_memory",
            "gpu_memory_bandwidth",
            "gpu_power",
            "gpu_flops",
            "gpu_nb_cores",
            "gpu_nb_tensor_cores",
            "params",
        ]
        dmatrix = xgb.DMatrix(features_input.reshape(1, -1), feature_names=feature_names)
        prediction = loaded_model.predict(dmatrix)
        if prediction.size > 0:
            return float(prediction[0]) / 1000  # Convert to GB
        else:
            print("[ERROR] Prediction returned empty array.")
            return 0.0


class UsageModel:
    def estimate_usage_cost(self, features_input: np.ndarray) -> float:
        """Predict execution time using xgboost model."""
        usage = features_input[0] / (features_input[1] * features_input[2])
        if usage:
            return float(usage) if float(usage) <  1.0 else 0.9
        else:
            print("[ERROR] Calculating Usage returned Invalid value.")
            return 0.0
