# entities/server.py
from simpy import Environment
from entities.processing_platform import ProcessingPlatform
from entities.location import Location


class Server:
    idx = 0

    def __init__(
        self,
        layer,
        location: Location,
        processing_platforms_list,
        deployed_models,
        ram_memory,
        bandwidth,
        env: Environment,
    ):
        Server.idx += 1
        self.id = Server.idx
        self.name = f"Server-{self.id}"
        self.layer = layer  # 'edge' or 'cloud'
        self.location = location
        self.processing_platforms_list = processing_platforms_list
        self.deployed_models = deployed_models
        self.ram_memory = ram_memory
        self.bandwidth = bandwidth
        self.env = env

    def setPPList(self, processing_platforms_list: list[ProcessingPlatform]):
        for pp in processing_platforms_list:
            if pp not in self.processing_platforms_list:
                pp.current_server = self
                self.processing_platforms_list.append(pp)
                print(
                    f"[INFO] Server-setPPList: Processing Unit {pp.name} added to Server {self.name}"
                )

    def setModelList(self, models: list):
        self.deployed_models = models
        for model in models:
            model.server = self
            print(f"[INFO] Server-setModelList: Model {model.name} deployed on Server {self.name}")
