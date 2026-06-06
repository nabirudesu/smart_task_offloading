# simulation_tracker.py - Comprehensive tracking system for edge computing simulation

import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
from entities.task import TaskStatus, Task
from entities.task_model_platform import TaskModelPlatform
from entities.servers.base_server import  Level

if TYPE_CHECKING:
    from entities.task_model_platform import ModelPlatformExecution
    from entities.servers.vehicle_server import VehicleServer  
    from entities.servers.base_server import BaseServer

class EventType(Enum):
    # Task Events
    TASK_GENERATION = "task_generation"
    TASK_SENDING = "task_sending"
    TASK_RECEIVING = "task_receiving"
    TASK_ADDED_TO_QUEUE = "task_added_to_queue"
    TASK_EXTRACTION = "task_extraction"
    TASK_EXECUTION_START = "task_execution_start"
    TASK_EXECUTION_END = "task_execution_end"
    
    # Cost Estimation Events
    COST_ESTIMATION_VEHICLE = "cost_estimation_vehicle"
    COST_ESTIMATION_EDGE = "cost_estimation_edge"
    COST_ESTIMATION_CLOUD = "cost_estimation_cloud"
    
    # Offloading Events
    OFFLOADING_INPUT_PREPARATION = "offloading_input_preparation"
    OFFLOADING_DECISION_CALCULATION = "offloading_decision_calculation"
    SENDING_OFFLOADING_DECISIONS = "sending_offloading_decisions"
    
    # Server Events
    SERVER_STATE_UPDATE = "server_state_update"
    VEHICLE_POSITION_UPDATE = "vehicle_position_update"
    
    # System Events
    SIMULATION_START = "simulation_start"
    SIMULATION_END = "simulation_end"

    # NEW: Handoff Events
    VEHICLE_HANDOFF = "vehicle_handoff"

# NEW: Handoff Snapshot
@dataclass
class HandoffSnapshot:
    """Snapshot of vehicle handoff event"""
    vehicle_id: int
    from_edge_id: int
    to_edge_id: int
    timestamp: float
    position: List[float]  # Position at handoff time

@dataclass
class TaskSnapshot:
    """Snapshot of task state at a specific time"""
    task_id: int
    task_type: str
    status: str
    vehicle_id: int
    min_accuracy: float
    max_latency: float
    data_size: float
    data_size_output: float
    arrival_time: float
    execution_start_time: float
    execution_end_time: float
    position_in_edge_queue: Optional[int] = None
    arrival_time_to_edge_queue: Optional[float] = None
    extraction_time_to_edge_queue: Optional[float] = None
    chosen_execution_level: Optional[str] = None
    chosen_execution_platform: Optional[str] = None
    chosen_execution_model: Optional[str] = None
    chosen_execution_memory_consumption :float = 0.0
    chosen_execution_execution_time :float = 0.0
    chosen_execution_energy_consumption :float = 0.0
    chosen_execution_platform_usage :float = 0.0
    chosen_execution_model_accuracy :float = 0.0
@dataclass 
class ServerSnapshot:
    """Snapshot of server state at a specific time"""
    server_id: int
    server_name: str
    server_type: str  # "vehicle", "edge", "cloud"
    level: str
    ram_capacity: float
    available_ram: float
    bandwidth: float
    route_plan: Optional[List] =None
    # Vehicle specific
    position: Optional[List[float]] = None
    speed: Optional[float] = None
    trip_finished: Optional[bool] = None
    power_of_vehicle: Optional[float] = None
    # Edge specific
    queue_size: Optional[int] = None
    queue_capacity: Optional[int] = None
    coverage_length: Optional[float] = None
    # Platform states
    platform_states: Optional[Dict[str, Dict]] = None
@dataclass
class EventRecord:
    """Record of a simulation event"""
    timestamp: float
    event_type: str
    entity_id: Optional[int] = None
    entity_type: Optional[str] = None  # "task", "server", "system"
    details: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None

class SimulationTracker:
    """Comprehensive tracker for edge computing simulation"""
    
    def __init__(self, output_dir: str = "simulation_output"):
        self.output_dir = output_dir
        self.events: List[EventRecord] = []
        self.task_snapshots: dict[int,TaskSnapshot] = {}
        self.server_snapshots: dict[int,ServerSnapshot] = {}
        self.handoff_snapshots: Dict[int, List[HandoffSnapshot]] = {}  # vehicle_id -> list of handoffs

        # Track active entities
        self.tasks: Dict[int, Task] = {}
        self.servers: Dict[int, "BaseServer"] = {}
        
        # Statistics
        self.stats = {
            "total_tasks_generated": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_events": 0,
            "simulation_start_time": None,
            "simulation_end_time": None,
            "total_handpffs":0,
        }
        
        print(f"[TRACKER] Simulation tracker initialized, output directory: {output_dir}")

    def register_server(self, server: "BaseServer"):
        """Register a server for tracking"""
        self.servers[server.id] = server
        self._log_event(
            event_type=EventType.SERVER_STATE_UPDATE,
            entity_id=server.id,
            entity_type="server",
            details={"action": "registered", "server_type": server.__class__.__name__}
        )

    def register_task(self, task: Task):
        """Register a task for tracking"""
        self.tasks[task.id] = task
        
    def _log_event(self, event_type: EventType, timestamp: Optional[float] = None,
                   entity_id: Optional[int] = None, entity_type: Optional[str] = None,
                   details: Optional[Dict[str, Any]] = None, 
                   performance_metrics: Optional[Dict[str, float]] = None):
        """Log a simulation event"""
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        event = EventRecord(
            timestamp=timestamp,
            event_type=event_type.value,
            entity_id=entity_id,
            entity_type=entity_type,
            details=details or {},
            performance_metrics=performance_metrics or {}
        )
        
        self.events.append(event)
        self.stats["total_events"] += 1
        
        print(f"[TRACKER] Event logged: {event_type.value} at {timestamp:.3f}s")

    def track_task_generation(self, task: Task, vehicle_id: int, timestamp: float):
        """Track task generation event"""
        self.register_task(task)
        self.stats["total_tasks_generated"] += 1
        
        self._log_event(
            event_type=EventType.TASK_GENERATION,
            timestamp=timestamp,
            entity_id=task.id,
            entity_type="task",
            details={
                "task_type": task.type,
                "vehicle_id": vehicle_id,
                "min_accuracy": task.min_accuracy,
                "max_latency": task.max_latency,
                "data_size": task.data_size
            }
        )
        
        # Take task snapshot
        self._take_task_snapshot(task,vehicle_id, timestamp)

    def track_task_sending(self, task_id: int, source_id: int, destination_id: int, 
                          timestamp: float, data_size: float):
        """Track task sending event"""
        self._log_event(
            event_type=EventType.TASK_SENDING,
            timestamp=timestamp,
            entity_id=task_id,
            entity_type="task",
            details={
                "source_id": source_id,
                "destination_id": destination_id,
                "data_size": data_size
            }
        )

    def track_task_receiving(self, task_id: int, receiver_id: int, timestamp: float):
        """Track task receiving event"""
        self._log_event(
            event_type=EventType.TASK_RECEIVING,
            timestamp=timestamp,
            entity_id=task_id,
            entity_type="task",
            details={"receiver_id": receiver_id}
        )

    def track_task_added_to_queue(self, task_id: int, server_id: int, position: int, timestamp: float):
        """Track task added to queue event"""
        self._log_event(
            event_type=EventType.TASK_ADDED_TO_QUEUE,
            timestamp=timestamp,
            entity_id=task_id,
            entity_type="task",
            details={
                "server_id": server_id,
                "queue_position": position
            }
        )

    def track_task_extraction(self, task_id: int, server_id: int, timestamp: float):
        """Track task extraction from queue event"""
        self._log_event(
            event_type=EventType.TASK_EXTRACTION,
            timestamp=timestamp,
            entity_id=task_id,
            entity_type="task",
            details={"server_id": server_id}
        )

    def track_cost_estimation(self, level: Level, task_id: int, timestamp: float, 
                             execution_details: List[Dict]):
        """Track cost estimation for a specific level"""
        event_map = {
            Level.DEVICE: EventType.COST_ESTIMATION_VEHICLE,
            Level.EDGE: EventType.COST_ESTIMATION_EDGE,
            Level.CLOUD: EventType.COST_ESTIMATION_CLOUD
        }
        
        self._log_event(
            event_type=event_map[level],
            timestamp=timestamp,
            entity_id=task_id,
            entity_type="task",
            details={
                "level": level.name,
                "execution_count": len(execution_details),
                "execution_details": execution_details
            }
        )

    def track_offloading_input_preparation(self, server_id: int, timestamp: float, 
                                         task_count: int, preparation_time: float):
        """Track offloading input preparation event"""
        self._log_event(
            event_type=EventType.OFFLOADING_INPUT_PREPARATION,
            timestamp=timestamp,
            entity_id=server_id,
            entity_type="server",
            details={"task_count": task_count},
            performance_metrics={"preparation_time": preparation_time}
        )
        print("[LOGGED ONCE MORE]")

    def track_offloading_decision_calculation(self, server_id: int, timestamp: float,
                                            task_count: int, calculation_time: float, 
                                            algorithm_used: str,tasks_with_decisions:list[TaskModelPlatform]):
        """Track offloading decision calculation event"""
        self._log_event(
            event_type=EventType.OFFLOADING_DECISION_CALCULATION,
            timestamp=timestamp,
            entity_id=server_id,
            entity_type="server",
            details={
                "task_count": task_count,
                "algorithm_used": algorithm_used
            },
            performance_metrics={"calculation_time": calculation_time}
        )
        print("**********************************")
        # print(len(tasks_with_decisions))
        for task in tasks_with_decisions:
            self._update_task_snapshot(task.task, task.chosen_execution)

    def track_sending_offloading_decisions(self, server_id: int, timestamp: float,
                                         vehicle_decisions: Dict[int, int]):
        """Track sending offloading decisions to vehicles"""
        self._log_event(
            event_type=EventType.SENDING_OFFLOADING_DECISIONS,
            timestamp=timestamp,
            entity_id=server_id,
            entity_type="server",
            details={
                "vehicle_count": len(vehicle_decisions),
                "vehicle_decisions": vehicle_decisions
            }
        )

    def track_task_execution_start(self, task_id: int, server_id: int, timestamp: float,
                                  execution: "ModelPlatformExecution"):
        """Track task execution start"""
        self._log_event(
            event_type=EventType.TASK_EXECUTION_START,
            timestamp=timestamp,
            entity_id=task_id,
            entity_type="task",
            details={
                "server_id": server_id,
                "model_name": execution.model.name if execution.model else None,
                "platform_name": execution.platform.name if execution.platform else None,
                "expected_execution_time": execution.execution_time,
                "memory_consumption": execution.memory_consumption,
                "energy_consumption": execution.energy_consumption,
                "platform_usage": execution.platform_usage
            }
        )
        
        # Update task in snapshot
        if task_id in self.tasks:
            self._update_task_snapshot(self.tasks[task_id],execution)

    def track_task_execution_end(self, task_id: int, server_id: int, timestamp: float,
                                final_status: TaskStatus, actual_execution_time: float):
        """Track task execution end"""
        if final_status == TaskStatus.SUCCESS:
            self.stats["tasks_completed"] += 1
        elif final_status in [TaskStatus.FAILED, TaskStatus.FAILED_RESOURCE_UNAVAILABLE]:
            self.stats["tasks_failed"] += 1
            
        self._log_event(
            event_type=EventType.TASK_EXECUTION_END,
            timestamp=timestamp,
            entity_id=task_id,
            entity_type="task",
            details={
                "server_id": server_id,
                "final_status": final_status.name
            },
            performance_metrics={"actual_execution_time": actual_execution_time}
        )
        
        # Update task in snapshot
        if task_id in self.tasks:
            self._update_task_snapshot_at_end(self.tasks[task_id])

    def track_server_state_update(self, server: "BaseServer", timestamp: float):
        """Track server state update"""
        self._log_event(
            event_type=EventType.SERVER_STATE_UPDATE,
            timestamp=timestamp,
            entity_id=server.id,
            entity_type="server",
            details={"server_type": server.__class__.__name__}
        )
        
        # Take server snapshot
        self._take_server_snapshot(server, timestamp)

    # NEW: Track vehicle handoff
    def track_vehicle_handoff(self, vehicle_id: int, from_edge_id: int, to_edge_id: int, timestamp: float, position: List[float]):
        """Track vehicle handoff event"""
        self._log_event(EventType.VEHICLE_HANDOFF, 
                        timestamp=timestamp,
                        entity_id=vehicle_id,
                        details={
            "vehicle_id": vehicle_id,
            "from_edge_id": from_edge_id,
            "to_edge_id": to_edge_id,
            "position": position
        })
        if self.stats.get("total_handoffs") is None:
            self.stats["total_handoffs"] = 1
        else:
            self.stats["total_handoffs"] += 1
        
        handoff = HandoffSnapshot(vehicle_id, from_edge_id, to_edge_id, timestamp, position)
        print("HANDOFF TRACKED")
        if vehicle_id not in self.handoff_snapshots:
            self.handoff_snapshots[vehicle_id] = []
        print("HANDOFF TRACKED1")
        self.handoff_snapshots[vehicle_id].append(handoff)
        print(f"[TRACKER] Vehicle {vehicle_id} handoff from Edge {from_edge_id} to {to_edge_id} at {timestamp}")


    def track_vehicle_position_update(self, vehicle: "VehicleServer", timestamp: float):
        """Track vehicle position update"""
        self._log_event(
            event_type=EventType.VEHICLE_POSITION_UPDATE,
            timestamp=timestamp,
            entity_id=vehicle.id,
            entity_type="server",
            details={
                "position": vehicle.position,
                "trip_finished": vehicle.trip_finished,
                "speed": vehicle.speed * 3.6  # Convert back to km/h
            }
        )

    def track_simulation_start(self, timestamp: float):
        """Track simulation start"""
        self.stats["simulation_start_time"] = timestamp
        self._log_event(
            event_type=EventType.SIMULATION_START,
            timestamp=timestamp,
            entity_type="system"
        )

    def track_simulation_end(self, timestamp: float):
        """Track simulation end"""
        self.stats["simulation_end_time"] = timestamp
        self._log_event(
            event_type=EventType.SIMULATION_END,
            timestamp=timestamp,
            entity_type="system"
        )

    def _take_task_snapshot(self, task: Task,server_id:int, timestamp: float):
        """Take a snapshot of task state"""
        # Get chosen execution details if available
        chosen_execution_level = None
        chosen_execution_platform = None
        chosen_execution_model = None
        
        if hasattr(task, 'task_model_platform') and task.task_model_platform:
            if hasattr(task.task_model_platform, 'chosen_execution') and task.task_model_platform.chosen_execution:
                chosen_execution = task.task_model_platform.chosen_execution
                chosen_execution_level = chosen_execution.level.name
                chosen_execution_platform = chosen_execution.platform.name if chosen_execution.platform else None
                chosen_execution_model = chosen_execution.model.name if chosen_execution.model else None
        
        snapshot = TaskSnapshot(
            task_id=task.id,
            task_type=task.type,
            status=task.status.name,
            vehicle_id=server_id,
            min_accuracy=task.min_accuracy,
            max_latency=task.max_latency,
            data_size=task.data_size,
            data_size_output=task.data_size_output,
            arrival_time=task.arrival_time,
            execution_start_time=task.execution_start_time,
            execution_end_time=task.execution_end_time,
            chosen_execution_level=chosen_execution_level,
            chosen_execution_platform=chosen_execution_platform,
            chosen_execution_model=chosen_execution_model
        )
        
        self.task_snapshots[task.id] = snapshot


    def _update_task_snapshot(self, task: Task,execution:"ModelPlatformExecution"):
        """Take a snapshot of task state"""
        # Get chosen execution details if available
        self.task_snapshots[task.id].chosen_execution_energy_consumption = execution.energy_consumption
        self.task_snapshots[task.id].chosen_execution_execution_time = execution.execution_time
        self.task_snapshots[task.id].chosen_execution_memory_consumption = execution.memory_consumption
        self.task_snapshots[task.id].chosen_execution_platform_usage = execution.platform_usage
        self.task_snapshots[task.id].chosen_execution_model_accuracy = execution.model_accuracy
        self.task_snapshots[task.id].chosen_execution_level = execution.level
        self.task_snapshots[task.id].chosen_execution_platform = execution.platform.name
        self.task_snapshots[task.id].chosen_execution_model = execution.model.name
        self.task_snapshots[task.id].execution_start_time = task.execution_start_time
        self.task_snapshots[task.id].execution_end_time = task.execution_end_time

    def _update_task_snapshot_at_end(self, task: Task):
        self.task_snapshots[task.id].execution_start_time = task.execution_start_time
        self.task_snapshots[task.id].execution_end_time = task.execution_end_time
        self.task_snapshots[task.id].status = task.status.name

    def _take_server_snapshot(self, server: "BaseServer", timestamp: float):
        """Take a snapshot of server state"""
        from entities.servers.vehicle_server import VehicleServer
        from entities.servers.edge_server import EdgeServer
        # Platform states
        platform_states = {}
        for platform in server.processing_platforms_list:
            platform_states[platform.name] = {
                "platform_usage": platform.platform_usage.level,
                "memory_size": platform.memory_size.level,
                "currently_executing": platform.currently_executing,
                "task_count": len(platform.task_list)
            }
        
        # Server-specific attributes
        server_specific = {}
        if isinstance(server, VehicleServer):
            server_specific.update({
                "position": server.position,
                "speed": server.speed * 3.6,  # Convert to km/h
                "trip_finished": server.trip_finished,
                "power_of_vehicle": server.power_of_vehicle,
                "route_plan":server.route_plan
            })
        elif isinstance(server, EdgeServer):
            server_specific.update({
                "position": [server.location.latitude,server.location.longitude],
                "queue_size": len(server.task_queue.items),
                "queue_capacity": server.task_queue.capacity,
                "coverage_length": server.length
            })
        
        snapshot = ServerSnapshot(
            server_id=server.id,
            server_name=server.name,
            server_type=server.__class__.__name__.lower().replace("server", ""),
            level=server.level.name,
            ram_capacity=server.ram_capacity,
            available_ram=server.available_ram,
            bandwidth=server.bandwidth,
            platform_states=platform_states,
            **server_specific
        )
        
        self.server_snapshots[server.id]=snapshot

    def export_to_files(self):
        """Export all tracking data to files"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Export events
        events_df = pd.DataFrame([asdict(event) for event in self.events])
        events_df.to_csv(f"{self.output_dir}/events.csv", index=False)
        
        # Export task snapshots
        tasks_df = pd.DataFrame([asdict(snapshot) for snapshot in self.task_snapshots.values()])
        tasks_df.to_csv(f"{self.output_dir}/task_snapshots.csv", index=False)
        
        # Export server snapshots
        servers_df = pd.DataFrame([asdict(snapshot) for snapshot in self.server_snapshots.values()])
        servers_df.to_csv(f"{self.output_dir}/server_snapshots.csv", index=False)
        
        # Export server snapshots
        handoff_df = pd.DataFrame([asdict(snapshot) for snapshot in self.handoff_snapshots.values()])
        handoff_df.to_csv(f"{self.output_dir}/handoff_snapshots.csv", index=False)

        # Export statistics
        with open(f"{self.output_dir}/simulation_stats.json", 'w') as f:
            json.dump(self.stats, f, indent=2)
            
        print(f"[TRACKER] Exported tracking data to {self.output_dir}/")

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the simulation"""
        duration = None
        if self.stats["simulation_start_time"] and self.stats["simulation_end_time"]:
            duration = self.stats["simulation_end_time"] - self.stats["simulation_start_time"]
            
        return {
            "simulation_duration": duration,
            "total_events": self.stats["total_events"],
            "total_tasks_generated": self.stats["total_tasks_generated"],
            "tasks_completed": self.stats["tasks_completed"],
            "tasks_failed": self.stats["tasks_failed"],
            "success_rate": self.stats["tasks_completed"] / max(1, self.stats["tasks_completed"] + self.stats["tasks_failed"]),
            "total_servers": len(self.servers),
            "total_task_snapshots": len(self.task_snapshots),
            "total_server_snapshots": len(self.server_snapshots),
            "total_handoff_snapshots": len(self.handoff_snapshots),
            "event_types": list(set(event.event_type for event in self.events))
        }

    def print_summary(self):
        """Print a summary of tracked data"""
        summary = self.get_summary_statistics()
        
        print("\n" + "="*60)
        print("SIMULATION TRACKING SUMMARY")
        print("="*60)
        print(f"Simulation Duration: {summary['simulation_duration']:.2f}s" if summary['simulation_duration'] else "Duration: Not available")
        print(f"Total Events Tracked: {summary['total_events']}")
        print(f"Total Tasks Generated: {summary['total_tasks_generated']}")
        print(f"Tasks Completed: {summary['tasks_completed']}")
        print(f"Tasks Failed: {summary['tasks_failed']}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print(f"Total Servers: {summary['total_servers']}")
        print(f"Task Snapshots: {summary['total_task_snapshots']}")
        print(f"Server Snapshots: {summary['total_server_snapshots']}")
        print(f"Event Types: {len(summary['event_types'])}")
        print("="*60)

# Global tracker instance
tracker: Optional[SimulationTracker] = None

def initialize_tracker(output_dir: str = "simulation_output") -> SimulationTracker:
    """Initialize the global tracker"""
    global tracker
    tracker = SimulationTracker(output_dir)
    return tracker

def get_tracker() -> Optional[SimulationTracker]:
    """Get the global tracker instance"""
    return tracker