# tracker.py
import json
from typing import Dict, List, Any, Optional,TYPE_CHECKING
from simpy import Environment

if TYPE_CHECKING:
    from entities.servers.base_server import BaseServer
    from entities.processing_platform import ProcessingPlatform

class Tracker:
    """
    Central tracker for simulation entities and events.
    Tracks tasks step-by-step and server states/actions.
    Collects data in memory and can dump to JSON.
    """
    def __init__(self, env: Environment):
        self.env = env
        self.tasks: Dict[int, List[Dict[str, Any]]] = {}  # task_id -> list of events
        self.servers: Dict[str, Dict[str, Any]] = {}  # server_id -> {state: {}, actions: []}
        self.global_events: List[Dict[str, Any]] = []  # For system-wide events

    def _get_timestamp(self) -> float:
        return self.env.now

    def track_task_event(self, task, event_type: str, details: Optional[Dict[str, Any]] = None):
        from entities.task import Task
        from entities.task_model_platform import TaskModelPlatform
        """Track a step/event for a specific task."""
        task_id = task.id if isinstance(task, Task) else task.task_id
        if task_id not in self.tasks:
            self.tasks[task_id] = []
        
        event = {
            "timestamp": self._get_timestamp(),
            "event_type": event_type,
            "status": task.status.name if isinstance(task, Task) else task.task.status.name,
            "details": details or {}
        }
        if isinstance(task, TaskModelPlatform) and task.chosen_execution:
            event["chosen_execution"] = {
                "level": task.chosen_execution.level.name,
                "model": task.chosen_execution.model.name if task.chosen_execution.model else None,
                "platform": task.chosen_execution.platform.name if task.chosen_execution.platform else None
            }
        self.tasks[task_id].append(event)

    def track_server_action(self, server: "BaseServer", action_type: str, details: Optional[Dict[str, Any]] = None):
        """Track an action performed by a server."""
        server_id = str(server.id)
        if server_id not in self.servers:
            self.servers[server_id] = {"state": {}, "actions": [], "type": server.level.name}
        
        action = {
            "timestamp": self._get_timestamp(),
            "action_type": action_type,
            "details": details or {}
        }
        self.servers[server_id]["actions"].append(action)
        # Update current state snapshot after action
        self._update_server_state(server)

    def _update_server_state(self, server: "BaseServer"):
        from entities.servers.vehicle_server import VehicleServer
        from entities.servers.edge_server import EdgeServer
        from entities.servers.cloud_server import CloudServer
        """Update the current state snapshot for a server."""
        server_id = str(server.id)
        state = {
            "timestamp": self._get_timestamp(),
            "location": {
                "city": server.location.city,
                "lat": server.location.latitude,
                "lon": server.location.longitude
            } if server.location else None,
            "ram": {
                "capacity": server.ram_capacity,
                "available": server.available_ram
            },
            "bandwidth": server.bandwidth,
            "processing_platforms": self._get_platforms_state(server.processing_platforms_list),
            "active_tasks": self._get_active_tasks(server),
            "connected_servers": self._get_connected_servers(server),
        }
        if isinstance(server, VehicleServer):
            state["vehicle_specific"] = {
                "position": server.position,
                "speed": server.speed,
                "trip_finished": server.trip_finished,
                "current_edge_server": server.current_edge_server,
                "power": server.power_of_vehicle
            }
        elif isinstance(server, EdgeServer):
            state["edge_specific"] = {
                "length": server.length,
                "queue_size": len(server.task_queue.items),
                "queue_capacity": server.task_queue.capacity,
                "power_e_e": server.power_P_e_e,
                "power_e_v": server.power_P_e_v,
                "power_e_c": server.power_P_e_c
            }
        elif isinstance(server, CloudServer):
            state["cloud_specific"] = {
                "power": server.power_P_c_e
            }
        self.servers[server_id]["state"] = state

    def _get_platforms_state(self, platforms: List["ProcessingPlatform"]) -> List[Dict[str, Any]]:
        """Get state of processing platforms."""
        return [
            {
                "name": p.name,
                "usage": p.platform_usage.level,
                "memory": p.memory_size.level,
                "ram": p.ram_size.level,
                "power_efficiency": p.power_efficiency.level,
                "currently_executing": p.currently_executing
            } for p in platforms
        ]

    def _get_active_tasks(self, server: "BaseServer") -> List[int]:
        """Get list of active task IDs on the server."""
        active = []
        for pp in server.processing_platforms_list:
            active.extend([t.id for t, _ in pp.task_list])
        return active

    def _get_connected_servers(self, server: "BaseServer") -> List[str]:
        from entities.servers.vehicle_server import VehicleServer
        from entities.servers.edge_server import EdgeServer
        from entities.servers.cloud_server import CloudServer

        """Get IDs of connected servers."""
        connected = []
        if isinstance(server, VehicleServer):
            if server.current_edge_server:
                connected.append(server.current_edge_server)
        elif isinstance(server, EdgeServer):
            if server.cloud_server:
                connected.append(str(server.cloud_server.id))
            connected.extend([str(v.id) for v in server.vehicles_servers_list or []])
        elif isinstance(server, CloudServer):
            # Cloud might connect to edges, but not explicitly tracked; add if needed
            pass
        return connected

    def track_global_event(self, event_type: str, details: Optional[Dict[str, Any]] = None):
        """Track system-wide events."""
        event = {
            "timestamp": self._get_timestamp(),
            "event_type": event_type,
            "details": details or {}
        }
        self.global_events.append(event)

    def dump_to_json(self, filename: str = "simulation_trace.json"):
        """Dump all tracked data to a JSON file."""
        data = {
            "tasks": self.tasks,
            "servers": self.servers,
            "global_events": self.global_events
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4, default=str)  # default=str for non-serializable types
        print(f"[TRACKER] Data dumped to {filename}")