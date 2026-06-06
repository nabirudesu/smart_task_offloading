# entities/server.py
from simpy import Environment
from entities.processing_platform import ProcessingPlatform
from entities.dnn_models import DnnModel
from entities.location import Location
from enum import Enum
from entities.cost_model import CostModel
from entities.task_model_platform import TaskModelPlatform, ModelPlatformExecution
from entities.task import TaskStatus, Task
from entities.tracker import Tracker

class Level(Enum):
    CLOUD = 2
    EDGE = 1
    DEVICE = 0

    def __str__(self):
        return self.name.lower()


class BaseServer:
    idx = 0

    def __init__(
        self,
        level: Level,
        location: Location,
        processing_platforms_list: list[ProcessingPlatform],
        deployed_models,
        ram_capacity,
        available_ram,
        bandwidth,
        # tracker: Tracker,
        env: Environment,
    ):
        BaseServer.idx += 1
        self.id = BaseServer.idx
        self.name = f"Server-{self.id}"
        self.level = level  # Default type, can be changed later
        self.location = location
        self.processing_platforms_list = processing_platforms_list
        self.deployed_models: list[DnnModel] = deployed_models
        self.ram_capacity = ram_capacity
        self.available_ram = available_ram  # Current
        self.bandwidth = bandwidth
        # self.tracker=tracker
        self.env = env

    def setPPList(self, processing_platforms_list: list[ProcessingPlatform]):
        for pp in processing_platforms_list:
            # if pp not in self.processing_platforms_list:
            pp.current_server = self
            # print(
            #     f"[INFO] Server-setPPList: Processing Unit {pp.name} added to Server {self.name}"
            # )

    def setModelList(self, models: list):
        self.deployed_models = models
        for model in models:
            model.server = self
            # print(f"[INFO] Server-setModelList: Model {model.name} deployed on Server {self.name}")

    def setCostModel(self, model_directory_path: str, layer: str) -> None:
        """
        Load the cost model for the vehicle layer.
        The model is expected to be in a specific directory structure.
        """
        self.cost_model: CostModel | None = None
        try:
            self.cost_model = CostModel(
                name=model_directory_path,
            )
            print(f"[INFO] Cost model loaded for {layer} layer: {self.cost_model.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load cost model: {e}")

    def start_task_execution(self, task: Task, execution: ModelPlatformExecution):
        if execution.model is None or execution.platform is None:
            print(
                f"[ERROR] Task {task.id} execution failed on {self.name}: Invalid model or platform"
            )
            task.status = TaskStatus.FAILED
            return
        if (
            execution.platform.platform_usage.level < execution.platform_usage
            or execution.platform.memory_size.level < execution.memory_consumption
        ):
            print(f"[ERROR] Task {task.id} failed on {self.name}: Insufficient resources")
            print(
                f"[DEBUG] PlatformUsage: {execution.platform.platform_usage.level} < {execution.platform_usage}, MemorySize: {execution.platform.memory_size.level} < {execution.memory_consumption}"
            )
            task.status = TaskStatus.FAILED_RESOURCE_UNAVAILABLE
            return
        task.status = TaskStatus.IN_EXEC
        task.execution_start_time = self.env.now  # type: ignore
        print(
            f"TIME {self.env.now:.3f}: [INFO] Task {task.id} started execution on {self.name} with model {execution.model.name} on platform {execution.platform.name}"
        )
        # self.tracker.track_task_event(task, "execution_started", {"on_server": self.id, "model": execution.model.name, "platform": execution.platform.name})
        # self.tracker.track_server_action(self, "task_execution_start", {"task_id": task.id})
        
        yield self.env.process(execution.platform.execute_task(task, execution))
        
        task.execution_end_time = self.env.now  # type: ignore
        task.status = TaskStatus.SUCCESS if task.is_success() else TaskStatus.FAILED

        # self.tracker.track_task_event(task, "execution_finished", {"status": task.status.name, "latency": task.execution_end_time - task.execution_start_time})
        # self.tracker.track_server_action(self, "task_execution_end", {"task_id": task.id, "status": task.status.name})

        print(f"TIME {self.env.now:.3f}: [INFO] Task {task.id} finished execution on {self.name}, status: {task.status}")
        print(f"TIME {self.env.now:.3f}: [INFO] Task {task.id} chosen execution cost : execution time:{execution.execution_time}, memory consumption:{execution.memory_consumption} utilization percentage:{execution.platform_usage} energy consumption{execution.energy_consumption}")
        print(f"TIME {self.env.now:.3f}: [INFO] Task {task.id} constraints: max_latency:{task.max_latency} start time {task.execution_start_time} end time {task.execution_end_time} arrival time {task.arrival_time}")