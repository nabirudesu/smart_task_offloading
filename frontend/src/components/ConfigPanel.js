import React, { useState, useEffect } from 'react';
import './ConfigPanel.css';

function ConfigPanel({
    configOptions,
    onStartSimulation,
    onStopSimulation,
    simulationStatus,
    currentConfig,
    connected
}) {
    const [config, setConfig] = useState({
        scenario: 'general',
        vehicles: 5,
        duration: 1,
        task_rate: 3,
        fps: 5,
        vehicle_speed: 50,
        edge_coverage: 1.0,
        city: 'paris',
        edge_servers: 1 // Default value for edge_servers
    });

    const [validationErrors, setValidationErrors] = useState({});

    useEffect(() => {
        // Reset validation errors when config changes
        setValidationErrors({});
    }, [config]);

    const handleInputChange = (field, value) => {
        setConfig(prev => ({
            ...prev,
            [field]: value
        }));
    };

    const validateConfig = () => {
        const errors = {};

        if (config.vehicles < 1 || config.vehicles > 50) {
            errors.vehicles = 'Vehicles must be between 1 and 50';
        }

        if (config.duration < 1 || config.duration > 5) {
            errors.duration = 'Duration ' + config.duration + 'must be between 1 and 5 seconds';
        }

        if (config.task_rate < 1 || config.task_rate > 5) {
            errors.task_rate = 'Vehicle Task rate must be between 1 and 5 tasks/Frame';
        }

        if (config.fps < 1 || config.fps > 20) {
            errors.fps = 'FPS must be between 1 and 20';
        }

        if (config.vehicle_speed < 10 || config.vehicle_speed > 250) {
            errors.vehicle_speed = 'Vehicle speed must be between 10 and 200 km/h';
        }

        if (config.edge_coverage < 0.01 || config.edge_coverage > 10) {
            errors.edge_coverage = 'Edge coverage must be between 0.5 and 10 km';
        }

        if (config.scenario === 'mobility' && (config.edge_servers < 1 || config.edge_servers > 5)) {
            errors.edge_servers = 'Edge servers must be between 2 and 4';
        }

        setValidationErrors(errors);
        return Object.keys(errors).length === 0;
    };

    const handleStart = () => {
        if (validateConfig()) {
            onStartSimulation(config);
        }
    };

    const isStartDisabled = () => {
        return !connected || simulationStatus === 'starting' || simulationStatus === 'running';
    };

    const isStopDisabled = () => {
        return simulationStatus !== 'running';
    };

    if (!configOptions) {
        return (
            <div className="config-panel">
                <h3>⚙️ Configuration</h3>
                <div className="loading">Loading configuration options...</div>
            </div>
        );
    }

    return (
        <div className="config-panel">
            <h3>⚙️ Simulation Configuration</h3>

            {/* Connection Status */}
            <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
                <span className="status-icon">{connected ? '🟢' : '🔴'}</span>
                <span>{connected ? 'Connected to Server' : 'Disconnected'}</span>
            </div>

            {/* Basic Configuration */}
            <div className="config-section">
                <h4>📍 Location & Scenario</h4>

                <div className="config-group">
                    <label htmlFor="city">City:</label>
                    <select
                        id="city"
                        value={config.city}
                        onChange={(e) => handleInputChange('city', e.target.value)}
                        disabled={simulationStatus === 'running'}
                    >
                        {configOptions.cities?.map(city => (
                            <option key={city.value} value={city.value}>
                                {city.label}
                            </option>
                        ))}
                    </select>
                    {config.city && configOptions.cities && (
                        <div className="config-info">
                            {configOptions.cities.find(c => c.value === config.city)?.center && (
                                <small>
                                    📍 {configOptions.cities.find(c => c.value === config.city).center[0].toFixed(4)}°,
                                    {configOptions.cities.find(c => c.value === config.city).center[1].toFixed(4)}°
                                </small>
                            )}
                        </div>
                    )}
                </div>

                <div className="config-group">
                    <label htmlFor="scenario">Scenario:</label>
                    <select
                        id="scenario"
                        value={config.scenario}
                        onChange={(e) => handleInputChange('scenario', e.target.value)}
                        disabled={simulationStatus === 'running'}
                    >
                        {configOptions.scenarios?.map(scenario => (
                            <option key={scenario.value} value={scenario.value}>
                                {scenario.label}
                            </option>
                        ))}
                    </select>
                    <div className="config-info">
                        <small>
                            {config.scenario === 'general'
                                ? '📊 Single edge server, basic setup'
                                : '🚛 Multiple edge servers, mobility testing'}
                        </small>
                    </div>
                </div>
            </div>

            {/* Simulation Parameters */}
            <div className="config-section">
                <h4>⏱️ Simulation Parameters</h4>

                <div className="config-group">
                    <label htmlFor="duration">Duration:</label>
                    <select
                        id="duration"
                        value={config.duration}
                        onChange={(e) => handleInputChange('duration', parseInt(e.target.value))}
                        disabled={simulationStatus === 'running'}
                    >
                        {configOptions.duration?.map(duration => (
                            <option key={duration.value} value={duration.value}>
                                {duration.label}
                            </option>
                        ))}
                    </select>
                    {validationErrors.duration && (
                        <div className="validation-error">{validationErrors.duration}</div>
                    )}
                </div>

                <div className="config-group">
                    <label htmlFor="vehicles">Number of Vehicles:</label>
                    <select
                        id="vehicles"
                        value={config.vehicles}
                        onChange={(e) => handleInputChange('vehicles', parseInt(e.target.value))}
                        disabled={simulationStatus === 'running'}
                    >
                        {configOptions.vehicles?.map(vehicle => (
                            <option key={vehicle.value} value={vehicle.value}>
                                {vehicle.label}
                            </option>
                        ))}
                    </select>
                    {validationErrors.vehicles && (
                        <div className="validation-error">{validationErrors.vehicles}</div>
                    )}
                </div>
            </div>

            {/* Advanced Parameters */}
            <div className="config-section">
                <h4>🔧 Advanced Parameters</h4>

                <div className="config-group">
                    <label htmlFor="task_rate">Task Generation Rate:</label>
                    <select
                        id="task_rate"
                        value={config.task_rate}
                        onChange={(e) => handleInputChange('task_rate', parseInt(e.target.value))}
                        disabled={simulationStatus === 'running'}
                    >
                        {configOptions.task_rate?.map(rate => (
                            <option key={rate.value} value={rate.value}>
                                {rate.label}
                            </option>
                        ))}
                    </select>
                    {validationErrors.task_rate && (
                        <div className="validation-error">{validationErrors.task_rate}</div>
                    )}
                </div>

                <div className="config-group">
                    <label htmlFor="fps">Vehicle FPS:</label>
                    <select
                        id="fps"
                        value={config.fps}
                        onChange={(e) => handleInputChange('fps', parseInt(e.target.value))}
                        disabled={simulationStatus === 'running'}
                    >
                        {configOptions.fps?.map(fps => (
                            <option key={fps.value} value={fps.value}>
                                {fps.label}
                            </option>
                        ))}
                    </select>
                    {validationErrors.fps && (
                        <div className="validation-error">{validationErrors.fps}</div>
                    )}
                    <div className="config-info">
                        <small>🎬 How often vehicles capture frames for processing</small>
                    </div>
                </div>

                <div className="config-group">
                    <label htmlFor="vehicle_speed">Vehicle Speed:</label>
                    <select
                        id="vehicle_speed"
                        value={config.vehicle_speed}
                        onChange={(e) => handleInputChange('vehicle_speed', parseInt(e.target.value))}
                        disabled={simulationStatus === 'running'}
                    >
                        {configOptions.vehicle_speed?.map(speed => (
                            <option key={speed.value} value={speed.value}>
                                {speed.label}
                            </option>
                        ))}
                    </select>
                    {validationErrors.vehicle_speed && (
                        <div className="validation-error">{validationErrors.vehicle_speed}</div>
                    )}
                </div>

                <div className="config-group">
                    <label htmlFor="edge_coverage">Edge Server Coverage:</label>
                    <select
                        id="edge_coverage"
                        value={config.edge_coverage}
                        onChange={(e) => handleInputChange('edge_coverage', parseFloat(e.target.value))}
                        disabled={simulationStatus === 'running'}
                    >
                        {configOptions.edge_coverage?.map(coverage => (
                            <option key={coverage.value} value={coverage.value}>
                                {coverage.label}
                            </option>
                        ))}
                    </select>
                    {validationErrors.edge_coverage && (
                        <div className="validation-error">{validationErrors.edge_coverage}</div>
                    )}
                    <div className="config-info">
                        <small>📡 Coverage radius for edge server communication</small>
                    </div>
                </div>

                {config.scenario === 'mobility' && (
                    <div className="config-group">
                        <label htmlFor="edge_servers">Number of Edge Servers:</label>
                        <select
                            id="edge_servers"
                            value={config.edge_servers}
                            onChange={(e) => handleInputChange('edge_servers', parseInt(e.target.value))}
                            disabled={simulationStatus === 'running'}
                        >
                            {configOptions.edge_servers?.map(server => (
                                <option key={server.value} value={server.value}>
                                    {server.label}
                                </option>
                            ))}
                        </select>
                        {validationErrors.edge_servers && (
                            <div className="validation-error">{validationErrors.edge_servers}</div>
                        )}
                        <div className="config-info">
                            <small>🌐 Number of edge servers for mobility scenario</small>
                        </div>
                    </div>
                )}
            </div>

            {/* Current Configuration Summary */}
            {currentConfig && simulationStatus === 'running' && (
                <div className="config-section current-config">
                    <h4>🔄 Active Configuration</h4>
                    <div className="config-summary">
                        <div className="summary-item">
                            <span className="label">City:</span>
                            <span className="value">{currentConfig.city}</span>
                        </div>
                        <div className="summary-item">
                            <span className="label">Vehicles:</span>
                            <span className="value">{currentConfig.vehicles}</span>
                        </div>
                        <div className="summary-item">
                            <span className="label">Duration:</span>
                            <span className="value">{currentConfig.duration}s</span>
                        </div>
                        <div className="summary-item">
                            <span className="label">Task Rate:</span>
                            <span className="value">{currentConfig.task_rate}/s</span>
                        </div>
                        {currentConfig.scenario === 'mobility' && (
                            <div className="summary-item">
                                <span className="label">Edge Servers:</span>
                                <span className="value">{currentConfig.edge_servers}</span>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Control Buttons */}
            <div className="control-buttons">
                <button
                    onClick={handleStart}
                    disabled={isStartDisabled()}
                    className={`control-button start-button ${isStartDisabled() ? 'disabled' : ''}`}
                    title={isStartDisabled() ? 'Cannot start simulation' : 'Start real simulation'}
                >
                    {simulationStatus === 'starting' ? '⏳ Starting...' : '🚀 Start Simulation'}
                </button>

                <button
                    onClick={onStopSimulation}
                    disabled={isStopDisabled()}
                    className={`control-button stop-button ${isStopDisabled() ? 'disabled' : ''}`}
                    title={isStopDisabled() ? 'No simulation running' : 'Stop current simulation'}
                >
                    ⏹️ Stop Simulation
                </button>
            </div>

            {/* Status Messages */}
            {simulationStatus === 'error' && (
                <div className="status-message error">
                    ❌ Simulation error occurred. Check console for details.
                </div>
            )}

            {simulationStatus === 'completed' && (
                <div className="status-message success">
                    ✅ Simulation completed successfully!
                </div>
            )}

            {!connected && (
                <div className="status-message warning">
                    ⚠️ Not connected to simulation server. Check if the backend is running.
                </div>
            )}

            {/* Task Types Info */}
            <div className="config-section info-section">
                <h4>📋 Available Task Types</h4>
                <div className="task-types">
                    <div className="task-type">🎯 <strong>DO:</strong> Object Detection</div>
                    <div className="task-type">🏷️ <strong>CI:</strong> Classification</div>
                    <div className="task-type">✂️ <strong>S:</strong> Segmentation</div>
                    <div className="task-type">🔍 <strong>OT:</strong> Object Tracking</div>
                    <div className="task-type">🚦 <strong>TLD:</strong> Traffic Light Detection</div>
                </div>
            </div>
        </div>
    );
}

export default ConfigPanel;