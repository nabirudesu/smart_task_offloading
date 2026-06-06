# Edge Computing Simulation Dashboard

A real-time web application for visualizing and managing edge computing task offloading simulations. The application features a Flask backend with WebSocket support and a React frontend with interactive maps.

## Features

- **Real-time Visualization**: Live map showing vehicle positions, edge server coverage, and task flow
- **Two Simulation Scenarios**:
  - **General Scenario**: Up to 30 vehicles, 150 tasks, single edge server
  - **Mobility Test**: 2-3 vehicles, multiple edge servers, handoff demonstration
- **Interactive Configuration**: Dropdown-based configuration for all simulation parameters
- **Live Statistics**: Real-time stats for vehicles, tasks, edge servers, and handoffs
- **WebSocket Updates**: 100ms update frequency for smooth real-time visualization
- **Multi-city Support**: Predefined locations in Paris, Berlin, and Algiers

## Architecture

### Backend (Flask + WebSocket)
- **Flask API**: RESTful endpoints for simulation control
- **Flask-SocketIO**: WebSocket communication for real-time updates
- **SimPy Integration**: Wrapper around your existing simulation code
- **Configurable Scenarios**: City-specific edge server placements

### Frontend (React + Leaflet)
- **React**: Component-based UI with real-time state management
- **Leaflet + OpenStreetMap**: Interactive maps with custom markers and coverage areas
- **Socket.IO Client**: WebSocket client for receiving live updates
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### 1. Clone and Setup Backend

```bash
# Create project directory
mkdir edge_simulation_web
cd edge_simulation_web

# Create backend directory
mkdir backend
cd backend

# Copy your simulation files here (entities folder, etc.)
# Copy the provided backend files:
# - app.py
# - simulation_wrapper.py  
# - config.py
# - requirements.txt

# Install Python dependencies
pip install -r requirements.txt

# macOS note: XGBoost requires OpenMP
# If you are on macOS, install libomp before running the backend:
# brew install libomp

# Optional: install eventlet for more reliable Flask-SocketIO websocket support
# pip install eventlet

# Start Flask backend
PORT=5001 python app.py
```

> Note: The backend now listens on port `5001` by default and supports a `PORT` environment variable.

### 2. Setup Frontend

```bash
# From project root, create frontend
cd ..
npx create-react-app frontend
cd frontend

# Install additional dependencies
npm install leaflet react-leaflet socket.io-client

# Replace default files with provided:
# - src/App.js
# - src/App.css
# - src/index.js
# - src/components/Map.js
# - src/components/ConfigPanel.js
# - src/components/StatsPanel.js
# - public/index.html
# - package.json

# Start React development server
npm start
```

### 3. Access Application

- **Backend API**: http://localhost:5001
- **Frontend Dashboard**: http://localhost:3000
- **WebSocket**: ws://localhost:5001

## Docker Deployment

This repository now includes Docker support for both backend and frontend.

### Build and run with Docker Compose

```bash
cd /Users/kyorakuna/Desktop/New folder/ad_sim
docker compose build
docker compose up
```

### Services
- `backend` available on `http://localhost:5001`
- `frontend` available on `http://localhost:3000`

### Environment variables
- `PORT` for backend port (default `5001`)
- `FRONTEND_ORIGIN` for backend CORS origin (default `http://localhost:3000`)

### Notes
- Backend output directories are mounted from `./backend/simulation_output`.
- Legacy duplicate files have been renamed with an `rm_` prefix instead of being deleted.

## Project Structure

```
edge_simulation_web/
├── backend/
│   ├── app.py                      # Flask application and API routes
│   ├── simulation_wrapper.py       # Integration wrapper for your simulation
│   ├── config.py                   # Configuration and city definitions
│   ├── requirements.txt            # Python dependencies
│   └── entities/                   # Your existing simulation classes
│       ├── servers/
│       ├── location.py
│       ├── network.py
│       └── ...
├── frontend/
│   ├── public/
│   │   └── index.html              # HTML template with Leaflet CSS
│   ├── src/
│   │   ├── components/
│   │   │   ├── Map.js              # Interactive map with real-time updates
│   │   │   ├── ConfigPanel.js      # Simulation configuration interface
│   │   │   └── StatsPanel.js       # Live statistics display
│   │   ├── App.js                  # Main React application
│   │   ├── App.css                 # Styling and responsive design
│   │   └── index.js                # React entry point
│   └── package.json                # Frontend dependencies
└── README.md                       # This file
```

## Configuration Options

### Scenarios
- **General**: Single edge server, up to 30 vehicles, focus on scalability
- **Mobility**: Multiple edge servers, up to 3 vehicles, focus on handoffs

### Parameters
- **Vehicles**: 1, 3, 5, 10, 15, 20, 30
- **Tasks per Frame**: 1, 2, 3, 4, 5
- **FPS**: 2, 4, 10, 15, 20
- **Cities**: Paris, Berlin, Algiers

### Edge Server Configuration
- **General Scenario**: 1 edge server with 2km coverage
- **Mobility Scenario**: 2-3 edge servers with 1.5km coverage each

## API Endpoints

### REST API
- `GET /api/health` - Health check
- `GET /api/config/options` - Get configuration options
- `POST /api/simulation/start` - Start simulation
- `POST /api/simulation/stop` - Stop simulation
- `GET /api/simulation/status` - Get simulation status

### WebSocket Events
- `connect` - Client connection established
- `subscribe_updates` - Subscribe to real-time updates
- `simulation_update` - Real-time simulation data (every 100ms)
- `simulation_ended` - Simulation completion notification

## Real-time Data Format

The WebSocket sends simulation updates every 100ms with this structure:

```json
{
  "timestamp": 15.3,
  "simulation_id": "abc123",
  "vehicles": [
    {
      "id": 1,
      "name": "Vehicle-1",
      "position": [48.8566, 2.3522],
      "destination": {"name": "Louvre", "lat": 48.8606, "lng": 2.3376},
      "total_tasks": 45,
      "tasks_completed": 32,
      "current_task_count": 3,
      "current_edge_server": 1,
      "trip_finished": false
    }
  ],
  "edge_servers": [
    {
      "id": 1,
      "name": "Paris Central Edge",
      "position": [48.8566, 2.3522],
      "coverage_radius": 2.0,
      "queue_size": 5,
      "queue_capacity": 10,
      "active_tasks": 3,
      "total_processed": 128
    }
  ],
  "cloud_server": {
    "id": "cloud-1",
    "name": "Cloud Server",
    "active_tasks": 12,
    "total_processed": 1024,
    "ram_usage": 0.3,
    "cpu_usage": 0.25
  },
  "scenario": "mobility",
  "city": "paris"
}
```

## Integration with Your Simulation

The `simulation_wrapper.py` currently contains mock simulation logic. To integrate your full simulation:

1. **Replace Mock Logic**: In `WebSimulation.setup_mock_simulation()`, replace with your actual simulation setup
2. **Update Processes**: Modify the SimPy processes to use your existing simulation logic  
3. **Data Mapping**: Ensure your simulation data matches the expected WebSocket format
4. **Import Classes**: Add your simulation classes to the import statements

Example integration points:
```python
# In simulation_wrapper.py
from your_simulation_module import EdgeComputingSimulation

class WebSimulation:
    def setup_simulation(self):
        # Use your existing simulation setup
        self.simulation = EdgeComputingSimulation()
        self.simulation.setup_simulation()
        
    def run_simulation(self):
        # Run your simulation with WebSocket emissions
        # Add self.emit_simulation_update() calls
```

## Customization

### Adding New Cities
Edit `backend/config.py` and add city configurations:

```python
CITY_CONFIGS = {
    "new_city": {
        "center": {"lat": XX.XXXX, "lng": X.XXXX, "name": "City Center"},
        "destinations": [...],
        "edge_servers": {
            "general": [...],
            "mobility": [...]
        }
    }
}
```

### Modifying Update Frequency
Change `UPDATE_FREQUENCY_MS` in `backend/config.py` (default: 100ms)

### Styling
Modify `frontend/src/App.css` for custom styling and themes

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure Flask-CORS is configured correctly in `app.py`
2. **WebSocket Connection Failed**: Check that Flask-SocketIO server is running on port 5000
3. **Map Not Loading**: Verify Leaflet CSS is included in `public/index.html`
4. **Marker Icons Missing**: Check that Leaflet marker icons are properly configured

### Debug Mode
- Backend: Set `debug=True` in `socketio.run()`
- Frontend: Use browser developer tools to monitor WebSocket messages

## Performance Optimization

- **Update Frequency**: Adjust based on system performance (100ms default)
- **Batch Size**: Modify `MAX_BATCH_SIZE` in simulation wrapper
- **Map Rendering**: Limit visible markers for large vehicle counts
- **Data Compression**: Consider compressing WebSocket messages for large datasets

## Future Enhancements

- [ ] Task flow visualization with animated lines
- [ ] Historical data playback and analysis
- [ ] Multiple simulation comparison
- [ ] Export simulation results to CSV/JSON
- [ ] Advanced handoff visualization with sequence diagrams
- [ ] Real-time performance metrics dashboard
- [ ] Scenario templates and presets
- [ ] Integration with cloud deployment platforms

## Support

For issues or questions:
1. Check the browser console for error messages
2. Verify all dependencies are installed correctly
3. Ensure backend simulation files are properly integrated
4. Test API endpoints directly using curl or Postman

This dashboard provides a solid foundation for visualizing edge computing simulations with room for extensive customization based on your specific research needs.