import math
import random
import simpy
from simpy import Environment, Store
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from entities.servers.base_server import BaseServer
URLLC_BANDWIDTH = 100e6  # 5G URLLC bandwidth in bps (100 Mbps)
URLLC_LATENCY = 4  # Average 5G URLLC latency in ms per hop


class Network:
    def __init__(self, env:Environment):
        self.env = env
        # self.downlink_bandwidth = URLLC_BANDWIDTH  # Downlink bandwidth in bps
        # self.uplink_bandwidth = URLLC_BANDWIDTH  # Uplink bandwidth in bps
        # self.signal_power = 1.0  # Signal power in Watts
        # self.interference_power = 0.1  # Interference power in Watts
        # self.noise_power = 0.01  # Noise power in Watts
        self.vehicle_to_edge_throughput = 77  # Vehicle to edge throughput in bps
        self.edge_to_edge_throughput = 700  # Edge to edge throughput in bps
        self.edge_to_cloud_throughput = 400  # Edge to cloud throughput in bps
        self.edge_to_vehicle_throughput = 550  # Edge to vehicle throughput in bps
        self.cloud_to_edge_throughput = 600  # Cloud to edge throughput in bps

    def transmit(self, data_size: int, source: "BaseServer", destination: "BaseServer"):
        """Simulate data transmission with appropriate delays."""
        from entities.servers.edge_server import EdgeServer
        from entities.servers.cloud_server import CloudServer
        from entities.servers.vehicle_server import VehicleServer

        if isinstance(source, VehicleServer):
            if isinstance(destination,EdgeServer):
                transmission_time = (data_size * 8) / self.vehicle_to_edge_throughput   
            else:
                transmission_time = (data_size * 8) / self.vehicle_to_edge_throughput + (data_size * 8) / self.edge_to_cloud_throughput  
        elif isinstance(source,EdgeServer):  
                if isinstance(destination,EdgeServer):
                    transmission_time = (data_size * 8) / self.edge_to_edge_throughput   
                elif isinstance(destination,CloudServer):
                    transmission_time = (data_size * 8) / self.edge_to_cloud_throughput   
                else :
                    transmission_time = (data_size * 8) / self.edge_to_vehicle_throughput 
        else:
            transmission_time = (data_size*8) / self.cloud_to_edge_throughput    
        transmission_time *= random.uniform(0.95, 1.05)  # Add variation

        yield self.env.timeout(transmission_time)
