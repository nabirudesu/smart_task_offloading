from entities.task import Task
from typing import Optional, TYPE_CHECKING
from datetime import datetime, timedelta


if TYPE_CHECKING:
    from entities.servers.base_server import Level
    from entities.dnn_models import DnnModel
    from entities.processing_platform import ProcessingPlatform
    from entities.servers.vehicle_server import VehicleServer


class ModelPlatformExecution:


    def __init__(
        self,
        level: "Level",
        task_id: Optional[int] = None,
        model: Optional["DnnModel"] = None,
        platform: Optional["ProcessingPlatform"] = None,
    ) -> None:
        if task_id is None or model is None or platform is None:
            return

        self.id = f"{task_id}_{model.id}_{platform.id}"
        self.task_id = task_id
        self.model = model
        self.platform = platform
        self.model_accuracy = model.accuracy
        self.model_type = model.type
        self.model_input_shape = model.input_shape
        self.level = level.__str__()
        self.id: str
        self.model: "DnnModel"
        self.platform: "ProcessingPlatform"
        self.model_accuracy: float
        self.model_type: str
        self.model_input_shape: tuple[float]
        self.alpha: float = 0.0  # Populated by edge if distance at future position > edge Length
        self.is_calculated: bool = False  # Set to True when cost model is calculated
        self.execution_time = 0.0  # Populated by CostModel
        self.memory_consumption = 0.0  # Populated by CostModel
        self.energy_consumption = 0.0  # Populated by CostModel
        self.platform_usage = 0.0  # Populated by CostModel
        self.duration_for_future_position: timedelta = timedelta(0)  # Populated by edge
        self.future_position: tuple[float, float] = (0.0, 0.0)  # Populated by edge


class TaskModelPlatform:

    def __init__(self, task: Task, execution: ModelPlatformExecution, vehicle: "VehicleServer"):
        self.task_id = task.id
        self.vehicle = vehicle
        self.task = task  # Keep full objects for flexibility
        self.position_in_edge_queue: Optional[int]
        self.future_position: tuple[float]
        self.duration_to_futur_position: timedelta
        self.arrival_time_to_edge_queue: datetime
        self.extraction_time_to_edge_queue: datetime
        self.executions_dict: dict[str, tuple[dict[str, bool], dict[tuple, ModelPlatformExecution]]] = {
            "V": ({"is_calculated": False}, {}),  # Vehicle
            "E": ({"is_calculated": False}, {}),  # Edge  
            "C": ({"is_calculated": False}, {}),  # Cloud
        }
        self.chosen_execution: Optional[
            ModelPlatformExecution
        ]  # Populated by edge after offloading decisions
        self.append_execution(execution)

    def append_execution(self, execution: ModelPlatformExecution):
        from entities.servers.base_server import Level
        if execution.level == Level.DEVICE.__str__():
            self.executions_dict["V"][1][execution.model.id, execution.platform.id] = execution
        elif execution.level == Level.EDGE.__str__():
            self.executions_dict["E"][1][execution.model.id, execution.platform.id] = execution
        else:
            self.executions_dict["C"][1][execution.model.id, execution.platform.id] = execution

    def set_chosen_execution(self, model_id, platform_id):
        for executions_dict in self.executions_dict.values():
            self.chosen_execution = executions_dict[1].get((model_id, platform_id))
            if self.chosen_execution:
                break
