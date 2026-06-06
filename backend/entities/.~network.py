import math
import random
import simpy
from simpy import Environment, Store

URLLC_BANDWIDTH = 100e6  # 5G URLLC bandwidth in bps (100 Mbps)
URLLC_LATENCY = 4  # Average 5G URLLC latency in ms per hop


class Network:
    def __init__(self, env:Environment):
        self.env = env
        self.downlink_bandwidth = URLLC_BANDWIDTH  # Downlink bandwidth in bps
        self.uplink_bandwidth = URLLC_BANDWIDTH  # Uplink bandwidth in bps
        self.signal_power = 1.0  # Signal power in Watts
        self.interference_power = 0.1  # Interference power in Watts
        self.noise_power = 0.01  # Noise power in Watts
        self.vehicle_to_edge_throughput = 100e6  # Vehicle to edge throughput in bps
        self.edge_to_edge_throughput = 100e6  # Edge to edge throughput in bps
        self.edge_to_cloud_throughput = 100e6  # Edge to cloud throughput in bps
        self.edge_to_vehicle_throughput = 100e6  # Edge to vehicle throughput in bps
        self.cloud_to_edge_throughput = 100e6  # Cloud to edge throughput in bps
        self.base_latency = URLLC_LATENCY

    # def __init__(
    #     self,
    #     env: simpy.Environment,
    #     downlink_bandwidth,
    #     uplink_bandwidth,
    #     signal_power,
    #     interference_power,
    #     noise_power,
    # ):
    #     self.env = env
    #     self.downlink_bandwidth = downlink_bandwidth  # Hz
    #     self.uplink_bandwidth = uplink_bandwidth  # Hz
    #     self.signal_power = signal_power  # Watts
    #     self.interference_power = interference_power  # Watts
    #     self.noise_power = noise_power  # Watts
    #     self.base_latency = URLLC_LATENCY

    def calculate_sinr(self):
        return self.signal_power / (self.interference_power + self.noise_power)

    def calculate_data_rate(self, sinr, bandwidth):
        return bandwidth * math.log2(1 + sinr)

    def transmission_delay(self, data_size):
        return data_size / self.downlink_bandwidth

    def transmit(self, data_size: int, source: str, destination: str):
        """Simulate data transmission with appropriate delays."""
        transmission_time = (data_size * 8) / self.uplink_bandwidth * 1000  # Convert to ms
        total_delay = self.base_latency + transmission_time
        total_delay *= random.uniform(0.9, 1.1)  # Add variation

        yield self.env.timeout(total_delay)
