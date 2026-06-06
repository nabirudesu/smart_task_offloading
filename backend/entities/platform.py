# entities/processing_platform.py
import simpy
import math
from entities.task import Task


class ProcessingPlatform:
    idx = 0

    def __init__(
        self,
        env: simpy.Environment,
        name,
        peak_flops,
        processing_frequency,
        processing_units,
        ram_size,
        memory_size,
        memory_bandwidth,
        memory_cache,
        power_efficiency,
        current_vehicle=None,
        current_server=None,
        num_tensor_cores=0.0,
    ):
        self.current_vehicle = current_vehicle
        self.current_server = current_server
        ProcessingPlatform.idx += 1
        self.id = ProcessingPlatform.idx
        self.name = f"{self.__class__.__name__}-{self.id}"
        self.env = env
        self.task_list: list = []
        self.status = 0.0
        self.peak_flops = peak_flops  # FLOPS
        self.processing_frequency = processing_frequency  # Hz
        self.memory_bandwidth = memory_bandwidth  # Bytes/second
        self.ram_size = ram_size  # Bytes
        self.num_tensor_cores = num_tensor_cores
        self.processing_units = simpy.Container(
            env, init=processing_units, capacity=processing_units
        )
        self.memory_size = simpy.Container(env, init=memory_size, capacity=memory_size)
        self.memory_cache = simpy.Container(env, init=memory_cache, capacity=memory_cache)
        self.power_efficiency = simpy.Container(
            env, init=power_efficiency, capacity=power_efficiency
        )
        self.currently_executing = 0

    def execute_task(self, task: Task, cost_model):

        req_units = self.processing_units.request()
        req_mem = self.memory_size.request()
        req_cache = self.memory_cache.request()
        req_power = self.power_efficiency.request()

        yield req_units & req_mem & req_cache & req_power

        self.currently_executing += 1
        self.task_list.append(task)
        self.update_status()

        print(
            f"[TIME {self.env.now:.2f}] Task {task.name} started execution on {self.name} in {self.current_server.layer if self.current_server else 'vehicle'} "
            f"with cost: {cost_model.cost:.2f}"
        )

        yield self.env.timeout(cost_model.execution_time)

        self.processing_units.release(req_units)
        self.memory_size.release(req_mem)
        self.memory_cache.release(req_cache)
        self.power_efficiency.release(req_power)

        self.task_list.remove(task)
        self.currently_executing -= 1
        self.update_status()

        print(
            f"[TIME {self.env.now:.2f}] Task {task.name} completed on {self.name} in {self.current_server.layer if self.current_server else 'vehicle'}"
        )

    def update_status(self):
        total_resources = {
            "processing_units": self.processing_units.capacity,
            "memory_size": self.memory_size.capacity,
            "power_efficiency": self.power_efficiency.capacity,
            "memory_cache": self.memory_cache.capacity,
        }
        used_resources = {
            "processing_units": self.processing_units.capacity - self.processing_units.level,
            "memory_size": self.memory_size.capacity - self.memory_size.level,
            "power_efficiency": self.power_efficiency.capacity - self.power_efficiency.level,
            "memory_cache": self.memory_cache.capacity - self.memory_cache.level,
        }
        utilization = {
            key: (used_resources[key] / total_resources[key]) * 100 for key in total_resources
        }
        self.status = sum(utilization.values()) / len(utilization)

        if self.currently_executing > 0:
            print(
                f"[STATUS] {self.name} in {self.current_server.layer if self.current_server else 'vehicle'}: {self.status:.2f}% overall utilization with {self.currently_executing} tasks running"
            )

    def get_real_task_execution_time(self, model_flops: float) -> float:
        return model_flops / self.peak_flops

    def get_task_energy_consumption(self, model_flops: float, model_required_power: float) -> float:
        return self.get_real_task_execution_time(model_flops) * model_required_power

    def get_resource_availability(self):
        return (
            self.processing_units.level / self.processing_units.capacity
            + self.memory_size.level / self.memory_size.capacity
            + self.memory_cache.level / self.memory_cache.capacity
            + self.power_efficiency.level / self.power_efficiency.capacity
        ) / 4.0


class CPU(ProcessingPlatform):
    def __init__(
        self,
        env,
        name,
        peak_flops,
        processing_frequency,
        processing_units,
        ram_size,
        memory_size,
        memory_bandwidth,
        memory_cache,
        power_efficiency,
        ISA,
    ):
        super().__init__(
            env,
            name,
            peak_flops,
            processing_frequency,
            processing_units,
            ram_size,
            memory_size,
            memory_bandwidth,
            memory_cache,
            power_efficiency,
        )
        self.ISA = ISA


class GPU(ProcessingPlatform):
    def __init__(
        self,
        env,
        name,
        peak_flops,
        processing_frequency,
        processing_units,
        ram_size,
        memory_size,
        memory_bandwidth,
        memory_cache,
        power_efficiency,
    ):
        super().__init__(
            env,
            name,
            peak_flops,
            processing_frequency,
            processing_units,
            ram_size,
            memory_size,
            memory_bandwidth,
            memory_cache,
            power_efficiency,
        )


class NPU(ProcessingPlatform):
    def __init__(
        self,
        env,
        name,
        peak_flops,
        processing_frequency,
        processing_units,
        ram_size,
        memory_size,
        memory_bandwidth,
        memory_cache,
        power_efficiency,
        AI_accelerator_units,
    ):
        super().__init__(
            env,
            name,
            peak_flops,
            processing_frequency,
            processing_units,
            ram_size,
            memory_size,
            memory_bandwidth,
            memory_cache,
            power_efficiency,
        )
        self.AI_accelerator_units = AI_accelerator_units


class TPU(ProcessingPlatform):
    def __init__(
        self,
        env,
        name,
        peak_flops,
        processing_frequency,
        processing_units,
        ram_size,
        memory_size,
        memory_bandwidth,
        memory_cache,
        power_efficiency,
    ):
        super().__init__(
            env,
            name,
            peak_flops,
            processing_frequency,
            processing_units,
            ram_size,
            memory_size,
            memory_bandwidth,
            memory_cache,
            power_efficiency,
        )
