# entities/edge_layer.py (Renamed from fog_layer.py)
from entities.server import Server


class EdgeLayer:
    def __init__(self, server_list: list):
        self.server_list = server_list

    def setServerList(self, server_list: list[Server]):
        for server in server_list:
            if server not in self.server_list:
                self.server_list.append(server)
                print(f"[INFO] Edge-Server List: Server {server.name} added to Edge")
