// src/UpdatedApp.js
// Updated React frontend with layout trigger handling and side-by-side display

import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import Map from './components/Map';
import ConfigPanel from './components/ConfigPanel';
import StatsPanel from './components/StatsPanel';
import TaskPanel from './components/TaskPanel';
import './App.css';

function UpdatedApp() {
    const [socket, setSocket] = useState(null);
    const [connected, setConnected] = useState(false);
    const [simulationData, setSimulationData] = useState(null);
    const [simulationStatus, setSimulationStatus] = useState('idle');
    const [configOptions, setConfigOptions] = useState(null);
    const [currentConfig, setCurrentConfig] = useState(null);
    const [simulationId, setSimulationId] = useState(null);
    const [connectionStatus, setConnectionStatus] = useState('disconnected');
    const [lastUpdateTime, setLastUpdateTime] = useState(null);
    const [showSimulation, setShowSimulation] = useState(false); // New state for layout control
    const [simPyTime, setSimPyTime] = useState(0.0); // New state for SimPy time

    useEffect(() => {
        // Initialize WebSocket connection
        const newSocket = io('http://localhost:5001', {
            transports: ['websocket', 'polling']
        });

        newSocket.on('connect', () => {
            console.log('Connected to updated simulation server');
            setConnected(true);
            setConnectionStatus('connected');
            newSocket.emit('subscribe_updates');
        });

        newSocket.on('disconnect', () => {
            console.log('Disconnected from simulation server');
            setConnected(false);
            setConnectionStatus('disconnected');
        });

        newSocket.on('connected', (data) => {
            console.log('Server welcome message:', data);
            setConnectionStatus('connected');
        });

        newSocket.on('subscribed', (data) => {
            console.log('Subscribed to updates:', data);
            setConnectionStatus('subscribed');
        });

        newSocket.on('simulation_update', (data) => {
            // Real-time updates from tracker with SimPy time synchronization
            setSimulationData(data);
            setSimPyTime(data.simpy_time || 0.0); // Update SimPy time
            console.log('Real simulation update:', data);
            setLastUpdateTime(new Date().toLocaleTimeString());
        });

        newSocket.on('simulation_ended', (data) => {
            console.log('Real simulation ended:', data);
            setSimulationStatus('completed');

            // Display final statistics
            if (data.final_stats) {
                console.log('Final Statistics:', data.final_stats);
            }
        });

        newSocket.on('simulation_error', (data) => {
            console.error('Simulation error:', data);
            setSimulationStatus('error');
        });

        // NEW: Handle layout change triggers
        newSocket.on('layout_change_trigger', (data) => {
            console.log('Layout change triggered:', data);
            setShowSimulation(data.show_simulation);
            setSimulationStatus('running');
            if (data.show_simulation) {
                // Simulation started, switch to side-by-side layout
                console.log('Switching to simulation view (50-50 layout)');
            } else {
                // Simulation stopped, back to config-only view
                console.log('Switching to configuration view');
                setSimulationData(null);
                setSimPyTime(0.0);
            }
        });

        setSocket(newSocket);

        // Fetch configuration options including all 15 cities
        fetchConfigOptions();

        return () => newSocket.close();
    }, []);

    const fetchConfigOptions = async () => {
        try {
            const response = await fetch('http://localhost:5001/api/config/options');
            const options = await response.json();
            console.log('Configuration options loaded:', options);
            setConfigOptions(options);
        } catch (error) {
            console.error('Failed to fetch config options:', error);
        }
    };

    const handleStartSimulation = async (config) => {
        try {
            setSimulationStatus('starting');
            console.log('Starting updated simulation with config:', config);

            const response = await fetch('http://localhost:5001/api/simulation/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config),
            });

            if (response.ok) {
                const result = await response.json();
                console.log('Updated simulation started:', result);
                setCurrentConfig(config);
                setSimulationId(result.simulation_id);
                setSimulationStatus('Completed');
                // Layout change will be triggered by WebSocket event
            } else {
                const error = await response.json();
                console.error('Failed to start simulation:', error);
                setSimulationStatus('error');
                alert(`Failed to start simulation: ${error.error}`);
            }
        } catch (error) {
            console.error('Error starting simulation:', error);
            setSimulationStatus('error');
            alert(`Error starting simulation: ${error.message}`);
        }
    };

    const handleStopSimulation = async () => {
        try {
            console.log('Stopping simulation...');
            const response = await fetch('http://localhost:5001/api/simulation/stop', {
                method: 'POST',
            });

            if (response.ok) {
                const result = await response.json();
                console.log('Simulation stopped:', result);
                setSimulationStatus('idle');
                setCurrentConfig(null);
                setSimulationId(null);
                // Layout change will be triggered by WebSocket event
            } else {
                const error = await response.json();
                console.error('Failed to stop simulation:', error);
            }
        } catch (error) {
            console.error('Error stopping simulation:', error);
        }
    };

    const handleExportData = async () => {
        if (!simulationId) {
            alert('No simulation data to export');
            return;
        }

        try {
            const response = await fetch(`http://localhost:5001/api/simulation/export/${simulationId}`);
            if (response.ok) {
                const exportInfo = await response.json();
                console.log('Export data available:', exportInfo);

                const filesList = exportInfo.export_files.map(f => f.name).join(', ');
                alert(`Export files available: ${filesList}\n\nFiles saved in: ${exportInfo.output_directory}`);
            } else {
                alert('No simulation data available for export');
            }
        } catch (error) {
            console.error('Error exporting data:', error);
            alert('Failed to export simulation data');
        }
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'running': return '#4CAF50';
            case 'starting': return '#FF9800';
            case 'completed': return '#2196F3';
            case 'error': return '#f44336';
            default: return '#9E9E9E';
        }
    };

    const getConnectionColor = (status) => {
        switch (status) {
            case 'connected': return '#4CAF50';
            case 'subscribed': return '#2196F3';
            default: return '#f44336';
        }
    };

    return (
        <div className="app">
            <header className="app-header">
                <h1>Edge Computing Simulation</h1>
                <div className="status-indicators">
                    <div
                        className="status-indicator"
                        style={{ backgroundColor: getConnectionColor(connectionStatus) }}
                    >
                        Connection: {connectionStatus}
                    </div>
                    <div
                        className="status-indicator"
                        style={{ backgroundColor: getStatusColor(simulationStatus) }}
                    >
                        Status: {simulationStatus}
                    </div>
                    {simPyTime > 0 && (
                        <div className="status-indicator" style={{ backgroundColor: '#2196F3' }}>
                            SimPy Time: {simPyTime.toFixed(3)}s
                        </div>
                    )}
                    {lastUpdateTime && (
                        <div className="status-indicator" style={{ backgroundColor: '#9E9E9E' }}>
                            Last Update: {lastUpdateTime}
                        </div>
                    )}
                </div>
            </header>

            <main className={`main-content ${showSimulation ? 'split-layout' : 'config-only'}`}>
                {/* Configuration Panel - Always visible */}
                <div className={`config-section ${showSimulation ? 'config-split' : 'config-full'}`}>
                    <ConfigPanel
                        connected={connected}
                        connectionStatus={connectionStatus}
                        simulationStatus={simulationStatus}
                        configOptions={configOptions}
                        currentConfig={currentConfig}
                        onStartSimulation={handleStartSimulation}
                        onStopSimulation={handleStopSimulation}
                        onExportData={handleExportData}
                    />

                    {showSimulation && (
                        <>
                            <StatsPanel
                                simulationData={simulationData}
                                simulationStatus={simulationStatus}
                                simPyTime={simPyTime}
                            />

                            <TaskPanel
                                simulationData={simulationData}
                                simulationStatus={simulationStatus}
                            />
                        </>
                    )}
                </div>

                {/* Simulation View - Only visible when simulation is running */}
                {showSimulation && (
                    <div className="simulation-section simulation-split">
                        {simulationData ? (
                            <>
                                <Map
                                    simulationData={simulationData}
                                    currentConfig={currentConfig}
                                    simPyTime={simPyTime}
                                />

                                <div className="simulation-info-panel">
                                    <h3>Live Simulation Data</h3>
                                    <div className="info-grid">
                                        <div className="info-item">
                                            <span className="info-label">Simulation ID:</span>
                                            <span className="info-value">{simulationId}</span>
                                        </div>
                                        <div className="info-item">
                                            <span className="info-label">SimPy Time:</span>
                                            <span className="info-value">{simPyTime.toFixed(3)}s</span>
                                        </div>
                                        <div className="info-item">
                                            <span className="info-label">City:</span>
                                            <span className="info-value">{currentConfig?.city || 'Unknown'}</span>
                                        </div>
                                        <div className="info-item">
                                            <span className="info-label">Scenario:</span>
                                            <span className="info-value">{currentConfig?.scenario || 'Unknown'}</span>
                                        </div>
                                        <div className="info-item">
                                            <span className="info-label">Vehicles:</span>
                                            <span className="info-value">{simulationData?.vehicles?.length || 0}</span>
                                        </div>
                                        <div className="info-item">
                                            <span className="info-label">Edge Servers:</span>
                                            <span className="info-value">{simulationData?.edge_servers?.length || 0}</span>
                                        </div>
                                    </div>

                                    {/* Cloud Server Info Panel */}
                                    {simulationData?.cloud_server && (
                                        <div className="cloud-info-section">
                                            <h4>☁️ Cloud Server Status</h4>
                                            <div className="cloud-stats">
                                                <div className="cloud-stat">
                                                    <span>Active Tasks:</span>
                                                    <span>{simulationData.cloud_server.active_tasks}</span>
                                                </div>
                                                <div className="cloud-stat">
                                                    <span>Total Processed:</span>
                                                    <span>{simulationData.cloud_server.total_processed}</span>
                                                </div>
                                                <div className="cloud-stat">
                                                    <span>RAM Usage:</span>
                                                    <span>{(simulationData.cloud_server.ram_usage * 100).toFixed(2)}%</span>
                                                </div>
                                                <div className="cloud-stat">
                                                    <span>Bandwidth:</span>
                                                    <span>{simulationData.cloud_server.bandwidth} Mbps</span>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </>
                        ) : (
                            <div className="simulation-placeholder">
                                <div className="placeholder-content">
                                    <h3>🚗 Real-time Simulation View</h3>
                                    <p>Waiting for simulation data...</p>
                                    <div className="loading-spinner"></div>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </main>

            {/* Footer */}
            <footer className="app-footer">
                <p>Edge Computing Simulation - Updated Integration v2.1</p>
                <p>Features: SimPy Time Sync | Proper Positioning | Layout Management</p>
            </footer>
        </div>
    );
}

export default UpdatedApp;
