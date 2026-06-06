import time

config = {
    "N_max": 150,
    "V_max": 30,
    "task_types": ["DO", "CI", "S", "OT", "TLD"],
    "hardware_names": ["NANO", "INTEL_I7", "TX2", "AGX"],
    "LEVELS": ["Vehicle", "Edge", "Cloud"],
    "Hardware_Vehicle": ["NANO", "INTEL_I7"],
    "Hardware_Edge": ["TX2", "NANO", "INTEL_I7"],
    "Hardware_Cloud": ["AGX", "NANO", "TX2", "INTEL_I7"],
    "MODEL_NAMES": [
        "SSD-MobileNetV3",
        "YOLOv5s",
        "SSD-EfficientNet",
        "YOLOv8x",
        "Mask R-CNN",
        "DETR",
        "MobileNetV3-Small",
        "EfficientNet-B0",
        "ResNet-18",
        "ResNet-50",
        "EfficientNet-B4",
        "ViT-Base",
        "MobileUnet",
        "DeepLabV3-MobileNet",
        "U-Net-Lite",
        "DeepLabV3-ResNet",
        "SegFormer",
        "TinySORT",
        "DeepSORT",
        "FairMOT-Lite",
        "FairMOT",
        "ByteTrack",
        "OC-SORT",
        "YOLOv3-Tiny-TLD",
        "YOLOv5s-TLD",
        "SSD-MobileNet-TLD",
        "YOLOv8m-TLD",
        "Faster R-CNN-TLD",
        "DETR-TLD",
    ],
    "Max_combinations_per_type": 20,
    "Max_combinations_per_vehicle": 2,
    "Max_combinations_per_edge": 6,
    "Max_combinations_per_cloud": 12,
    "max_steps": 10,
}

# Define normalization parameters and *ranges* for DataGenerator random values
norm_params: dict[str, dict] = {
    "min_accuracy": {"min": 0.0, "max": 1.0},
    "time_to_deadline": {"min": 0.0, "max": 3.0},  # Max latency in seconds
    "data_size_input": {"min": 0, "max": 120},  # MB
    "data_size_output": {"min": 0, "max": 100},  # MB
    "charge_remaining_percentage": {"min": 0.0, "max": 1.0},  # For hardware utilization capacity
    "memory_capacity_remaining_of_this_hardware": {
        "min": 0.0,
        "max": 128000.0,
    },  # MB, for specific hardware RAM capacity
    "power_P_v_e": {"min": 1.0, "max": 50.0},
    "power_P_e_e": {"min": 1.0, "max": 50.0},
    "power_P_e_v": {"min": 1.0, "max": 50.0},
    "power_P_e_c": {"min": 1.0, "max": 50.0},
    "power_P_c_e": {"min": 1.0, "max": 50.0},
    "vehicle_to_edge_throughput": {"min": 10.0, "max": 500.0},
    "edge_to_edge_throughput": {"min": 10.0, "max": 500.0},
    "edge_to_cloud_throughput": {"min": 10.0, "max": 500.0},
    "edge_to_vehicle_throughput": {"min": 10.0, "max": 500.0},
    "cloud_to_edge_throughput": {"min": 10.0, "max": 500.0},
    "execution_time": {"min": 0.0, "max": 10.0},
    "power_consumption": {"min": 1.0, "max": 50.0},
    "memory_consumption": {"min": 0.0, "max": 128000.0},  # MB
    "utilization_percentage": {"min": 0.0, "max": 1.0},
    "accuracy_of_model": {"min": 0.0, "max": 1.0},
}
