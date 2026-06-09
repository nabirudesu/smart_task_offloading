# Smart Task Offloading — Edge Computing Simulation

A real-time web application for visualizing and managing edge computing task offloading simulations. Vehicles generate DNN inference tasks that are offloaded across vehicle, edge, and cloud tiers using a PPO reinforcement learning agent.

The app is available in this link
https://crate-ivy-symptom.ngrok-free.dev

## Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask, Flask-SocketIO, SimPy, Ray RLlib |
| Frontend | React, Leaflet / OpenStreetMap, Socket.IO |
| Deployment | Docker Compose + Nginx |

## Features

- **Live map** — vehicle positions, edge server coverage zones, and task flow updated in real time
- **Two scenarios** — General (up to 30 vehicles, 1 edge server) and Mobility (2–4 edge servers, handoff demonstration)
- **PPO offloading agent** — trained RL model selects local / edge / cloud placement per task batch
- **Multi-city support** — city coordinates loaded from `backend/Data/locations.json`
- **WebSocket streaming** — simulation state pushed to the frontend every 100 ms

## Project Structure

```
smart_task_offloading/
├── backend/
│   ├── app.py                        # Flask app, API routes, SocketIO events
│   ├── integrated_simulation.py      # SimulationManager wiring Flask ↔ SimPy
│   ├── main_simulation.py            # EdgeComputingSimulation orchestrator
│   ├── config_data.py                # Scenario configuration helpers
│   ├── requirements.txt
│   ├── Data/
│   │   ├── locations.json            # City coordinates
│   │   ├── dnn_models.json           # DNN model specs per tier
│   │   └── processing_platforms.json # Hardware platform specs
│   ├── entities/
│   │   ├── servers/                  # VehicleServer, EdgeServer, CloudServer
│   │   ├── task.py / task_model_platform.py
│   │   ├── network.py / location.py
│   │   └── simulation_tracker_exteded.py
│   ├── ppo_rl_agent/                 # PPO environment, policy, training
│   ├── final_model/                  # Trained RLlib checkpoint
│   └── simulation_output/            # CSV / JSON results per run
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   └── components/               # Map, ConfigPanel, StatsPanel
│   ├── nginx.conf
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## Running with Docker (recommended)

```bash
git clone <repo-url>
cd smart_task_offloading
docker compose build
docker compose up
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:5001 |

**Environment variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5001` | Backend listen port |
| `HOST` | `0.0.0.0` | Backend bind address |
| `FRONTEND_ORIGIN` | `http://localhost:3000` | Allowed CORS origin |

## Running Manually

**Backend**

```bash
cd backend
pip install -r requirements.txt
# macOS: brew install libomp  (required for XGBoost)
python app.py
```

**Frontend**

```bash
cd frontend
npm install
npm start
```

## API Reference

### REST

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/config/options` | Available cities, scenarios, and parameters |
| `POST` | `/api/simulation/start` | Start a simulation |
| `POST` | `/api/simulation/stop` | Stop the active simulation |

**`POST /api/simulation/start` — required fields**

```json
{
  "scenario": "general",
  "vehicles": 5,
  "duration": 30,
  "city": "paris"
}
```

### WebSocket Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | server → client | Connection acknowledged |
| `subscribe_updates` | client → server | Subscribe to live updates |
| `simulation_update` | server → client | Full simulation state, every 100 ms |
| `simulation_ended` | server → client | Simulation completed |
| `layout_change_trigger` | server → client | Show / hide simulation view |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CORS error | Verify `FRONTEND_ORIGIN` matches the frontend URL |
| WebSocket fails | Confirm backend is reachable on port 5001 |
| Map tiles missing | Check internet access (tiles served by OpenStreetMap) |
| XGBoost import error on macOS | `brew install libomp` |
