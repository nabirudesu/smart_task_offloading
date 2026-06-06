# entities/cloud_layer.py
from entities.server import Server


class CloudLayer:
    def __init__(self, server_list: list):
        self.server_list = server_list

    def setServerList(self, server_list: list[Server]):
        for server in server_list:
            if server not in self.server_list:
                self.server_list.append(server)
                print(f"[INFO] Cloud-Server List: Server {server.name} added to Cloud")
