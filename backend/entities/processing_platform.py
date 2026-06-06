# entities/processing_platform.py
import simpy
from typing import Optional, TYPE_CHECKING

from entities.task import Task, TaskStatus

if TYPE_CHECKING:
    from entities.task_model_platform import ModelPlatformExecution
    from entities.servers.base_server import BaseServer
    from entities.servers.vehicle_server import VehicleServer


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
        self.current_server: Optional["BaseServer"] = current_server
        ProcessingPlatform.idx += 1
        self.id = ProcessingPlatform.idx
        self.id_name = f"{self.__class__.__name__}-{self.id}"
        self.name = name
        self.env = env
        self.task_list: list = []
        self.status = 0.0
        self.peak_flops = peak_flops  # FLOPS
        self.processing_frequency = processing_frequency  # Hz
        self.memory_bandwidth = memory_bandwidth  # Bytes/second
        # self.ram_size = ram_size  # Bytes
        self.num_tensor_cores = num_tensor_cores
        self.processing_units = simpy.Container(
            env, init=processing_units, capacity=processing_units
        )
        # self.memory_size = simpy.Container(env, init=memory_size, capacity=memory_size)
        self.memory_cache = simpy.Container(env, init=memory_cache, capacity=memory_cache)
        self.power_efficiency = simpy.Container(
            env, init=power_efficiency, capacity=power_efficiency
        )
        self.currently_executing = 0

        # Attributes used for tracking tasks execution (these are the ones used in the RL MODEL training)
        self.platform_usage = simpy.Container(env,init=1.0, capacity=1.0)  # Percentage of platform usage
        self.ram_size = simpy.Container(env,init=ram_size, capacity=ram_size)
        self.memory_size = simpy.Container(env,init=memory_size, capacity=memory_size)

    def execute_task(self, task: Task, execution: "ModelPlatformExecution"):
        """
        Execute a task on the platform, consuming shared resources and logging state.
        Args:
            task: Task object to execute.
            execution: ModelPlatformExecution with execution parameters.
        """
        # Required resources
        required_memory = execution.memory_consumption
        required_usage = execution.platform_usage
        # required_energy = execution.energy_consumption

        # Log initial resource state
        self._log_resource_state(task.id, "Before")

        try:
            # Request resources
            if isinstance(self, CPU)and self.current_server.available_ram < required_memory:
                if self.current_server.available_ram < required_memory:  # type: ignore
                    raise simpy.Interrupt("Insufficient RAM")
                self.current_server.available_ram -= required_memory  # type: ignore
            else:
                if self.memory_size.level < required_memory:
                    raise simpy.Interrupt("Insufficient MEMORY")
                else:
                    yield self.memory_size.get(required_memory)

            if self.platform_usage.level < required_usage:
                raise simpy.Interrupt("PROCESSING PLATFORM is FULL")
            else:
                yield self.platform_usage.get(required_usage)

            # # Update vehicle power if applicable
            # if isinstance(self.current_server, VehicleServer):
            #     self.current_server.power_of_vehicle -= required_energy
            #     self.current_server.power_of_vehicle = max(
            #         0.0, self.current_server.power_of_vehicle
            #     )

            # Track active task
            self.task_list.append((task, execution))
            self.currently_executing += 1

            # Log resource state during execution
            self._log_resource_state(task.id, "During")

            # Simulate execution
            yield self.env.timeout(execution.execution_time)

            # Update task status
            task.execution_end_time = self.env.now  # type: ignore
            task.status = TaskStatus.SUCCESS if task.is_success() else TaskStatus.FAILED
            print(f"TIME {self.env.now:.3f}: [INFO] Task {task.id} executed on {self.name}, status: {task.status}")

        except simpy.Interrupt as e:
            print(f"[ERROR] Task {task.id} failed on {self.name}: {e.cause}")
            task.status = TaskStatus.FAILED_RESOURCE_UNAVAILABLE
            return

        finally:
            # Release resources
            yield self.platform_usage.put(required_usage)
            if isinstance(self, CPU):
                self.current_server.available_ram += required_memory  # type: ignore
            else:
                yield self.memory_size.put(required_memory)
            # if isinstance(self.current_server, VehicleServer):
            #     self.current_server.power_of_vehicle += required_energy
            #     self.current_server.power_of_vehicle = max(
            #         0.0, self.current_server.power_of_vehicle
            #     )

            # Remove task and update status
            self.task_list.remove((task, execution))
            self.currently_executing -= 1
            self.update_status()
            self._log_resource_state(task.id, "After")

    def _log_resource_state(self, task_id: int, phase: str):
        from entities.servers.vehicle_server import VehicleServer
        """
        Log resource state for debugging and monitoring.
        """
        server = self.current_server
        log = (
            f"[INFO] {phase} Task {task_id} on {self.name}: "
            f"ProcessingUnits={self.processing_units.level}/{self.processing_units.capacity}, "
            f"Memory={self.memory_size.level}/{self.memory_size.capacity}, "
            f"Power={self.power_efficiency.level}/{self.power_efficiency.capacity}, "
            f"ServerRAM={server.available_ram}/{server.ram_capacity}, "
            f"PlatformUsage={self.platform_usage.level:.2f}%"
        )
        if isinstance(server, VehicleServer):
            log += f", VehiclePower={server.power_of_vehicle}"
        print(log)

    def update_status(self):
        """
        Update platform and server status based on active tasks.
        """
        # Update platform usage
        total_usage = sum(execution.platform_usage for _, execution in self.task_list)
        # self.platform_usage = total_usage / len(self.task_list) if self.task_list else 0.0

        # Update server attributes directly
        if self.current_server:
            # Track platform usage in server
            """
            # Optional feature, to store all platforms usages of the server in action
            if not hasattr(self.current_server, "platform_usages"):
                self.current_server.platform_usages = {
                    p.id: 0.0 for p in self.current_server.processing_platforms_list
                }
            self.current_server.platform_usages[self.id] = self.platform_usage
            """

            # Update vehicle power if applicable
            # if isinstance(self.current_server, VehicleServer):
            #     self.current_server.power_of_vehicle = max(
            #         0.0, self.current_server.power_of_vehicle
            #     )
            pass
        print(
            f"TIME {self.env.now:.3f}: [INFO] Platform {self.name} status updated: Usage={self.platform_usage.level:.2f}%, "
            f"ActiveTasks={len(self.task_list)}"
        )

    def get_real_task_execution_time(self, model_flops: float) -> float:
        return model_flops / self.peak_flops

    def get_task_energy_consumption(self, model_flops: float, model_required_power: float) -> float:
        return self.get_real_task_execution_time(model_flops) * model_required_power

    def get_resource_availability(self):
        pass


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
