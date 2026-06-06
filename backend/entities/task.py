# entities/task.py
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from entities.servers.vehicle_server import VehicleServer


class TaskStatus(Enum):
    CREATED = 0
    READY = 1
    QUEUED = 2
    IN_EXEC = 3
    SUCCESS = 4
    FAILED = 5
    CANCELED = 6
    PAUSED = 7
    RESUMED = 8
    FAILED_RESOURCE_UNAVAILABLE = 9


class Task:
    idx = 0

    def __init__(
        self,
        type,
        min_accuracy,
        max_latency,
        data_size,
        data_size_output,
        status,
        arrival_time,
        current_vehicle: Optional["VehicleServer"] = None,
    ):
        Task.idx += 1
        self.id = Task.idx
        self.vehicle = current_vehicle  # should add vehicule_id as in the RL model
        self.type = type  # should be integer as in the RL model
        self.min_accuracy = min_accuracy
        self.max_latency = max_latency  # transform it to time_to_deadline like in the RL model
        self.data_size = data_size  # this is euqivalent to the data_size_input in the RL model
        self.data_size_output = data_size_output
        self.status = status
        self.arrival_time = arrival_time
        self.execution_start_time = -1
        self.execution_end_time = -1

    def is_success(self):
        return self.is_started() and self.is_finished() and self.is_finished_before_deadline()

    def is_failed(self):
        return (self.is_started() and self.is_finished()) and not self.is_finished_before_deadline()

    def is_incomplete(self):
        return self.is_started() and not self.is_finished()

    def is_started(self):
        return self.execution_start_time != -1

    def is_finished(self):
        return self.execution_end_time != -1

    def is_finished_before_deadline(self):
        return self.execution_end_time <= self.arrival_time + self.max_latency

    def save_as_dict(self) -> dict:
        return {
            "id": self.id,
            "vehicle": self.vehicle,
            "task_type": self.type,
            "min_accuracy": self.min_accuracy,
            "max_latency": self.max_latency,
            "data_size": self.data_size,
            "status": self.status,
            "arrival_time": self.arrival_time,
            "execution_start_time": self.execution_start_time,
            "execution_end_time": self.execution_end_time,
        }
