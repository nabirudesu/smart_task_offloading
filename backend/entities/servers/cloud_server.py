from entities.servers.base_server import BaseServer, Level
from entities.location import Location
from entities.processing_platform import ProcessingPlatform
from simpy import Environment
from entities.dnn_models import DnnModel
# from entities.tracker import Tracker


class CloudServer(BaseServer):
    def __init__(
        self,
        level: Level,
        location: Location,
        processing_platforms_list: list[ProcessingPlatform],
        deployed_models: list[DnnModel],
        ram_capacity: float,
        avilable_ram: float,
        bandwidth: float,
        power_P_c_e: float,
        # tracker: Tracker,
        env: Environment,
    ):
        super().__init__(
            level,
            location,
            processing_platforms_list,
            deployed_models,
            ram_capacity,
            avilable_ram,
            bandwidth,
            # tracker,
            env,
        )
        self.name = f"Cloud-{self.id}"
        self.latency = 0.1  # Higher latency (seconds)
        self.high_capacity = True  # More resources
        self.power_P_c_e = power_P_c_e
        self.setCostModel("cloud_cost_model", "C")
