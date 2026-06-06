// src/components/StatsPanel.js
import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
    LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
    XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import './StatsPanel.css';

const MAX_DATA_POINTS = 200;

// Custom Tooltip Formatter
const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        return (
            <div className="custom-tooltip">
                <p className="label">{`Time : ${label}`}</p>
                {payload.map((pld, index) => (
                    <p key={index} style={{ color: pld.color }}>
                        {`${pld.name} : ${parseFloat(pld.value).toFixed(2)}`}
                    </p>
                ))}
            </div>
        );
    }
    return null;
};


function StatsPanel({ simulationData, simulationStatus, simPyTime }) {
    const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1', '#d0ed57', '#a4de6c'];

    // State for time-series and real-time server data
    const [timeSeriesData, setTimeSeriesData] = useState([]);
    const [vehicleServerState, setVehicleServerState] = useState([]);
    const [edgeServerState, setEdgeServerState] = useState([]);
    const [cloudServerState, setCloudServerState] = useState([]);
    const [selectedVehicle, setSelectedVehicle] = useState(null);
    const lastProcessedTime = useRef(null);

    // Main effect to process all incoming simulation data
    useEffect(() => {
        if (!simulationData || simPyTime === lastProcessedTime.current) {
            return;
        }

        // --- 1. Update Line Chart Data ---
        const processedTasks = simulationData.tasks.filter(t => t.status === 'SUCCESS');
        const calcAverage = (tasks, field) => {
            if (tasks.length === 0) return 0;
            const filteredTasks = tasks.filter(t => t[field] != null);
            if (filteredTasks.length === 0) return 0;
            const total = filteredTasks.reduce((sum, task) => sum + task[field], 0);
            return total / filteredTasks.length;
        };
        const newTimeSeriesPoint = {
            time: simPyTime,
            avgResponse: calcAverage(processedTasks, 'response_time'),
            avgEnergy: calcAverage(processedTasks, 'chosen_execution_energy_consumption'),
            avgMemory: calcAverage(processedTasks, 'chosen_execution_memory_consumption') / 1e6,
            avgUsage: calcAverage(processedTasks, 'chosen_execution_platform_usage') * 100,
            avgAccuracy: calcAverage(processedTasks, 'chosen_execution_model_accuracy') * 100,
        };

        setTimeSeriesData(prevData => {
            const updatedData = [...prevData, newTimeSeriesPoint];
            return updatedData.length > MAX_DATA_POINTS
                ? updatedData.slice(updatedData.length - MAX_DATA_POINTS)
                : updatedData;
        });

        // --- 2. Update Server State Bar Chart Data (Real-time) ---
        // Vehicle Server State
        if (selectedVehicle) {
            const vehicle = simulationData.vehicles.find(v => v.name === selectedVehicle);
            if (vehicle && vehicle.platform_states) {
                const vehicleState = Object.keys(vehicle.platform_states).map(platform => ({
                    name: platform,
                    usage: vehicle.platform_states[platform].platform_usage * 100,
                    tasks: vehicle.platform_states[platform].currently_executing,
                }));
                setVehicleServerState(vehicleState);
            }
        }

        // Edge Server State
        const edgeState = simulationData.edge_server_states?.map(server => ({
            name: `Edge ${server.id}`,
            cpu: server.cpu_usage * 100,
            ram: server.ram_usage * 100,
            disk: server.disk_usage * 100,
        })) || [];
        setEdgeServerState(edgeState);

        // Cloud Server State
        const cloud = simulationData.cloud_server_state;
        const cloudState = cloud ? [{
            name: 'Cloud Server',
            cpu: cloud.cpu_usage * 100,
            ram: cloud.ram_usage * 100,
            disk: cloud.disk_usage * 100,
        }] : [];
        setCloudServerState(cloudState);


        lastProcessedTime.current = simPyTime;

    }, [simulationData, simPyTime, selectedVehicle]);

    // Effect to set the default selected vehicle
    useEffect(() => {
        if (simulationData && simulationData.vehicles && simulationData.vehicles.length > 0 && !selectedVehicle) {
            setSelectedVehicle(simulationData.vehicles[0].name);
        }
    }, [simulationData, selectedVehicle]);


    // Memoized calculations for static pie charts (less frequent updates)
    const {
        taskStatusData,
        executionLevelData,
        platformDistributionData,
    } = useMemo(() => {
        if (!simulationData) {
            return { taskStatusData: [], executionLevelData: [], platformDistributionData: [] };
        }

        const statusCounts = simulationData.tasks.reduce((acc, task) => { acc[task.status] = (acc[task.status] || 0) + 1; return acc; }, {});
        const taskStatus = Object.keys(statusCounts).map(status => ({ name: status, value: statusCounts[status] }));

        const executionLevelCounts = simulationData.tasks.reduce((acc, task) => { const level = task.chosen_execution_level || 'Pending'; acc[level] = (acc[level] || 0) + 1; return acc; }, {});
        const executionLevel = Object.keys(executionLevelCounts).map(level => ({ name: level, value: executionLevelCounts[level] }));

        const platformCounts = simulationData.tasks.filter(t => t.chosen_execution_platform).reduce((acc, task) => { const platform = task.chosen_execution_platform; acc[platform] = (acc[platform] || 0) + 1; return acc; }, {});
        const platformDistribution = Object.keys(platformCounts).map(platform => ({ name: platform, value: platformCounts[platform] }));

        return {
            taskStatusData: taskStatus,
            executionLevelData: executionLevel,
            platformDistributionData: platformDistribution,
        };
    }, [simulationData]);


    if (!simulationData) {
        return <div className="stats-panel"><div className="placeholder-section"><h4>Waiting for simulation data...</h4></div></div>;
    }

    return (
        <div className="stats-panel">
            <h2>Simulation Statistics</h2>
            {/* --- REAL-TIME LINE CHARTS --- */}
            <div className="stats-section chart-row">
                <h3>Average Response Time</h3>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={timeSeriesData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(t) => t.toFixed(1)} label={{ value: 'Simulation Time (s)', position: 'insideBottom', offset: -5 }} />
                        <YAxis label={{ value: 'Time (s)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip content={<CustomTooltip />} />
                        <Legend />
                        <Line type="monotone" dataKey="avgResponse" name="Avg. Response (s)" stroke="#8884d8" strokeWidth={2} dot={false} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
            <div className="stats-section chart-row">
                <h3>Average Energy Consumption</h3>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={timeSeriesData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(t) => t.toFixed(1)} label={{ value: 'Simulation Time (s)', position: 'insideBottom', offset: -5 }} />
                        <YAxis label={{ value: 'Energy (J)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip content={<CustomTooltip />} />
                        <Legend />
                        <Line type="monotone" dataKey="avgEnergy" name="Avg. Energy (J)" stroke="#82ca9d" strokeWidth={2} dot={false} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
            <div className="stats-section chart-row">
                <h3>Average Memory Consumption</h3>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={timeSeriesData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(t) => t.toFixed(1)} label={{ value: 'Simulation Time (s)', position: 'insideBottom', offset: -5 }} />
                        <YAxis label={{ value: 'Memory (MB)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip content={<CustomTooltip />} />
                        <Legend />
                        <Line type="monotone" dataKey="avgMemory" name="Avg. Memory (MB)" stroke="#ffc658" strokeWidth={2} dot={false} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
            <div className="stats-section chart-row">
                <h3>Average Platform Usage</h3>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={timeSeriesData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(t) => t.toFixed(1)} label={{ value: 'Simulation Time (s)', position: 'insideBottom', offset: -5 }} />
                        <YAxis domain={[0, 100]} label={{ value: 'Usage (%)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip content={<CustomTooltip />} />
                        <Legend />
                        <Line type="monotone" dataKey="avgUsage" name="Avg. Usage (%)" stroke="#ff7c7c" strokeWidth={2} dot={false} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
            <div className="stats-section chart-row">
                <h3>Average Model Accuracy</h3>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={timeSeriesData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(t) => t.toFixed(1)} label={{ value: 'Simulation Time (s)', position: 'insideBottom', offset: -5 }} />
                        <YAxis domain={[0, 100]} label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip content={<CustomTooltip />} />
                        <Legend />
                        <Line type="monotone" dataKey="avgAccuracy" name="Avg. Accuracy (%)" stroke="#8dd1e1" strokeWidth={2} dot={false} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* --- PIE CHARTS (STATIC) --- */}
            <div className="stats-section">
                <div className="pie-charts-grid">
                    <div>
                        <h3>Task Status Distribution</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <PieChart>
                                <Pie data={taskStatusData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} fill="#8884d8" label>
                                    {taskStatusData.map((entry, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                    <div>
                        <h3>Execution Level Distribution</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <PieChart>
                                <Pie data={executionLevelData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} fill="#ff7c7c" label>
                                    {executionLevelData.map((entry, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                    <div>
                        <h3>Task Execution by Platform</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <PieChart>
                                <Pie data={platformDistributionData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} fill="#d0ed57" label>
                                    {platformDistributionData.map((entry, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default StatsPanel;