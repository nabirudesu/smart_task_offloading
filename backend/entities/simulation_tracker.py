# entities/simulation_tracker.py

import json
import csv
from typing import Dict, List, Any, Optional
from datetime import datetime
from entities.task import Task, TaskStatus
from entities.servers.base_server import BaseServer
from entities.servers.vehicle_server import VehicleServer
from entities.servers.edge_server import EdgeServer
from entities.servers.cloud_server import CloudServer

class SimulationTracker:
    """Comprehensive tracker for edge computing simulation events, tasks, and server states."""
    
    def __init__(self, env):
        self.env = env
        
        # Event logs
        self.task_events: List[Dict] = []
        self.server_states: List[Dict] = []
        self.simulation_events: List[Dict] = []
        
        # Tracking statistics
        self.total_tasks_generated = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        
        print(f"[TRACKER] Simulation tracker initialized at time {self.env.now:.2f}")

    def log_task_event(self, task_id: int, event_type: str, details: Dict = None, 
                      server_id: Optional[int] = None, server_type: Optional[str] = None):
        """Log task-related events throughout the task lifecycle."""
        event = {
            "timestamp": self.env.now,
            "task_id": task_id,
            "event_type": event_type,
            "server_id": server_id,
            "server_type": server_type,
            "details": details or {}
        }
        self.task_events.append(event)
        print(f"[TRACKER] Task Event: {event_type} - Task {task_id} at time {self.env.now:.3f}")

    def log_server_state(self, server: BaseServer, additional_info: Dict = None):
        """Log current state of a server (vehicle, edge, or cloud)."""
        server_type = "vehicle" if isinstance(server, VehicleServer) else \
                     "edge" if isinstance(server, EdgeServer) else \
                     "cloud" if isinstance(server, CloudServer) else "unknown"
        
        state = {
            "timestamp": self.env.now,
            "server_id": server.id,
            "server_type": server_type,
            "server_name": server.name,
            "ram_capacity": server.ram_capacity,
            "available_ram": server.available_ram,
            "bandwidth": server.bandwidth,
            "num_processing_platforms": len(server.processing_platforms_list),
            "num_deployed_models": len(server.deployed_models),
        }
        
        # Add server-specific information
        if isinstance(server, VehicleServer):
            state.update({
                "position": server.position,
                "trip_finished": server.trip_finished,
                "power_of_vehicle": server.power_of_vehicle,
                "current_edge_server": server.current_edge_server,
                "total_tasks": len(server.tasks_list),
                "tasks_to_be_sent": len(server.tasks_to_be_sent)
            })
        elif isinstance(server, EdgeServer):
            state.update({
                "queue_capacity": server.task_queue.capacity,
                "current_queue_size": len(server.task_queue.items),
                "length": server.length,
                "power_P_e_e": server.power_P_e_e,
                "power_P_e_v": server.power_P_e_v,
                "power_P_e_c": server.power_P_e_c,
                "num_vehicles": len(server.vehicles_servers_list) if server.vehicles_servers_list else 0
            })
        elif isinstance(server, CloudServer):
            state.update({
                "power_P_c_e": server.power_P_c_e,
                "high_capacity": server.high_capacity,
                "latency": server.latency
            })
        
        # Add processing platform states
        platform_states = []
        for platform in server.processing_platforms_list:
            platform_state = {
                "platform_id": platform.id,
                "platform_name": platform.name,
                "platform_usage": platform.platform_usage.level,
                "platform_capacity": platform.platform_usage.capacity,
                "memory_size": platform.memory_size.level,
                "memory_capacity": platform.memory_size.capacity,
                "currently_executing": platform.currently_executing,
                "active_tasks": len(platform.task_list)
            }
            platform_states.append(platform_state)
        
        state["processing_platforms"] = platform_states
        
        # Add additional info if provided
        if additional_info:
            state.update(additional_info)
        
        self.server_states.append(state)

    def log_simulation_event(self, event_type: str, details: Dict = None, 
                           entity_id: Optional[int] = None, entity_type: Optional[str] = None):
        """Log general simulation events."""
        event = {
            "timestamp": self.env.now,
            "event_type": event_type,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "details": details or {}
        }
        self.simulation_events.append(event)
        print(f"[TRACKER] Simulation Event: {event_type} at time {self.env.now:.3f}")

    # Specific event logging methods
    def log_task_generation(self, vehicle_id: int, task_types: List[str], num_tasks: int):
        """Log task generation event."""
        self.total_tasks_generated += num_tasks
        self.log_simulation_event(
            "task_generation",
            {
                "vehicle_id": vehicle_id,
                "task_types": task_types,
                "num_tasks_generated": num_tasks,
                "total_tasks_generated": self.total_tasks_generated
            },
            entity_id=vehicle_id,
            entity_type="vehicle"
        )

    def log_cost_estimation(self, entity_id: int, entity_type: str, task_id: int, 
                           level: str, execution_count: int = None):
        """Log cost estimation events."""
        self.log_simulation_event(
            f"cost_estimation_{level.lower()}",
            {
                "task_id": task_id,
                "estimation_level": level,
                "execution_combinations": execution_count
            },
            entity_id=entity_id,
            entity_type=entity_type
        )

    def log_task_sending(self, sender_id: int, receiver_id: int, task_id: int, 
                        data_size: int, transmission_time: float = None):
        """Log task sending events."""
        self.log_simulation_event(
            "task_sending",
            {
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "task_id": task_id,
                "data_size": data_size,
                "transmission_time": transmission_time
            },
            entity_id=sender_id,
            entity_type="vehicle"
        )

    def log_task_receiving(self, receiver_id: int, sender_id: int, task_id: int):
        """Log task receiving events."""
        self.log_simulation_event(
            "task_receiving",
            {
                "receiver_id": receiver_id,
                "sender_id": sender_id,
                "task_id": task_id
            },
            entity_id=receiver_id,
            entity_type="edge"
        )

    def log_task_queue_operation(self, server_id: int, operation: str, task_id: int, 
                                queue_size: int, position: int = None):
        """Log task queue operations (add/extract)."""
        self.log_simulation_event(
            f"task_queue_{operation}",
            {
                "server_id": server_id,
                "task_id": task_id,
                "queue_size": queue_size,
                "position_in_queue": position
            },
            entity_id=server_id,
            entity_type="edge"
        )

    def log_batch_extraction(self, server_id: int, batch_size: int, task_ids: List[int], 
                           remaining_tasks: int):
        """Log batch extraction from queue."""
        self.log_simulation_event(
            "batch_extraction",
            {
                "server_id": server_id,
                "batch_size": batch_size,
                "extracted_task_ids": task_ids,
                "remaining_in_queue": remaining_tasks
            },
            entity_id=server_id,
            entity_type="edge"
        )

    def log_offloading_preparation(self, server_id: int, num_tasks: int, 
                                 preparation_time: float):
        """Log offloading input preparation."""
        self.log_simulation_event(
            "offloading_inputs_preparation",
            {
                "server_id": server_id,
                "num_tasks": num_tasks,
                "preparation_time": preparation_time
            },
            entity_id=server_id,
            entity_type="edge"
        )

    def log_offloading_decision(self, server_id: int, num_tasks: int, algorithm: str, 
                              decision_time: float, decisions: List[Dict] = None):
        """Log offloading decision calculation."""
        self.log_simulation_event(
            "offloading_decision_calculation",
            {
                "server_id": server_id,
                "num_tasks": num_tasks,
                "algorithm": algorithm,
                "decision_time": decision_time,
                "decisions": decisions or []
            },
            entity_id=server_id,
            entity_type="edge"
        )

    def log_decision_sending(self, server_id: int, vehicle_id: int, num_decisions: int, 
                           transmission_time: float):
        """Log sending offloading decisions to vehicle."""
        self.log_simulation_event(
            "sending_offloading_decisions",
            {
                "server_id": server_id,
                "vehicle_id": vehicle_id,
                "num_decisions": num_decisions,
                "transmission_time": transmission_time
            },
            entity_id=server_id,
            entity_type="edge"
        )

    def log_task_execution_start(self, task_id: int, server_id: int, server_type: str, 
                                model_name: str, platform_name: str, execution_time: float):
        """Log task execution start."""
        self.log_task_event(
            task_id, 
            "execution_start",
            {
                "model_name": model_name,
                "platform_name": platform_name,
                "estimated_execution_time": execution_time
            },
            server_id=server_id,
            server_type=server_type
        )

    def log_task_execution_end(self, task_id: int, server_id: int, server_type: str, 
                              status: str, actual_execution_time: float):
        """Log task execution completion."""
        if status == "SUCCESS":
            self.total_tasks_completed += 1
        else:
            self.total_tasks_failed += 1
            
        self.log_task_event(
            task_id,
            "execution_end",
            {
                "status": status,
                "actual_execution_time": actual_execution_time,
                "total_completed": self.total_tasks_completed,
                "total_failed": self.total_tasks_failed
            },
            server_id=server_id,
            server_type=server_type
        )

    def log_task_status_change(self, task_id: int, old_status: str, new_status: str, 
                              server_id: int = None):
        """Log task status changes."""
        self.log_task_event(
            task_id,
            "status_change",
            {
                "old_status": old_status,
                "new_status": new_status
            },
            server_id=server_id
        )

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of the simulation."""
        return {
            "total_tasks_generated": self.total_tasks_generated,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "success_rate": self.total_tasks_completed / max(1, self.total_tasks_generated),
            "total_task_events": len(self.task_events),
            "total_server_states": len(self.server_states),
            "total_simulation_events": len(self.simulation_events),
            "simulation_duration": self.env.now
        }

    def export_logs(self, output_dir: str = "simulation_logs", prefix: str = "simulation"):
        """Export all logs to files."""
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export task events
        task_events_file = f"{output_dir}/{prefix}_task_events_{timestamp}.json"
        with open(task_events_file, "w") as f:
            json.dump(self.task_events, f, indent=2, default=str)
        
        # Export server states
        server_states_file = f"{output_dir}/{prefix}_server_states_{timestamp}.json"
        with open(server_states_file, "w") as f:
            json.dump(self.server_states, f, indent=2, default=str)
        
        # Export simulation events
        sim_events_file = f"{output_dir}/{prefix}_simulation_events_{timestamp}.json"
        with open(sim_events_file, "w") as f:
            json.dump(self.simulation_events, f, indent=2, default=str)
        
        # Export summary statistics
        summary_file = f"{output_dir}/{prefix}_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(self.get_summary_statistics(), f, indent=2, default=str)
        
        # Export CSV files for easier analysis
        self._export_csv_files(output_dir, prefix, timestamp)
        
        print(f"[TRACKER] Logs exported to {output_dir} with prefix {prefix}_{timestamp}")
        return {
            "task_events": task_events_file,
            "server_states": server_states_file,
            "simulation_events": sim_events_file,
            "summary": summary_file
        }

    def _export_csv_files(self, output_dir: str, prefix: str, timestamp: str):
        """Export logs as CSV files for analysis."""
        
        # Task events CSV
        if self.task_events:
            task_csv_file = f"{output_dir}/{prefix}_task_events_{timestamp}.csv"
            with open(task_csv_file, "w", newline="") as f:
                if self.task_events:
                    writer = csv.DictWriter(f, fieldnames=self.task_events[0].keys())
                    writer.writeheader()
                    for event in self.task_events:
                        # Flatten nested details
                        row = event.copy()
                        if isinstance(row.get("details"), dict):
                            for k, v in row["details"].items():
                                row[f"detail_{k}"] = v
                            del row["details"]
                        writer.writerow(row)
        
        # Simulation events CSV
        if self.simulation_events:
            sim_csv_file = f"{output_dir}/{prefix}_simulation_events_{timestamp}.csv"
            with open(sim_csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.simulation_events[0].keys())
                writer.writeheader()
                for event in self.simulation_events:
                    # Flatten nested details
                    row = event.copy()
                    if isinstance(row.get("details"), dict):
                        for k, v in row["details"].items():
                            row[f"detail_{k}"] = v
                        del row["details"]
                    writer.writerow(row)

    def start_periodic_server_monitoring(self, servers: List[BaseServer], interval: float = 1.0):
        """Start periodic monitoring of server states."""
        def monitor_servers():
            while True:
                for server in servers:
                    self.log_server_state(server)
                yield self.env.timeout(interval)
        
        self.env.process(monitor_servers())
        print(f"[TRACKER] Started periodic server monitoring every {interval}s")

    def print_simulation_summary(self):
        """Print a summary of tracked simulation data."""
        stats = self.get_summary_statistics()
        
        print("\n" + "="*60)
        print("SIMULATION TRACKING SUMMARY")
        print("="*60)
        print(f"Simulation Duration: {stats['simulation_duration']:.2f}s")
        print(f"Total Tasks Generated: {stats['total_tasks_generated']}")
        print(f"Total Tasks Completed: {stats['total_tasks_completed']}")
        print(f"Total Tasks Failed: {stats['total_tasks_failed']}")
        print(f"Success Rate: {stats['success_rate']:.2%}")
        print(f"Total Task Events Logged: {stats['total_task_events']}")
        print(f"Total Server State Snapshots: {stats['total_server_states']}")
        print(f"Total Simulation Events: {stats['total_simulation_events']}")
        print("="*60)