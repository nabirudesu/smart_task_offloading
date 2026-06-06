// TaskPanel component for displaying task execution details and statistics
import React from 'react';
import './TaskPanel.css';

function TaskPanel({ simulationData, simulationStatus }) {

    const getTasksByStatus = () => {
        if (!simulationData || !simulationData.tasks) {
            return { total: 0, completed: 0, failed: 0, active: 0, tasks: [] };
        }

        const tasks = simulationData.tasks;
        const completed = tasks.filter(task => task.status === 'SUCCESS').length;
        const failed = tasks.filter(task => task.status === 'FAILED' || task.status === 'FAILED_RESOURCE_UNAVAILABLE').length;
        const active = tasks.filter(task => task.status === 'IN_EXEC' || task.status === 'CREATED').length;

        return {
            total: tasks.length,
            completed,
            failed,
            active,
            tasks: tasks.slice(-10) // Show last 10 tasks
        };
    };

    const getTaskTypeDistribution = () => {
        if (!simulationData || !simulationData.tasks) {
            return {};
        }

        const distribution = {};
        simulationData.tasks.forEach(task => {
            distribution[task.type] = (distribution[task.type] || 0) + 1;
        });

        return distribution;
    };

    const getTaskTypeLabel = (type) => {
        const labels = {
            'DO': 'Object Detection',
            'CI': 'Classification',
            'S': 'Segmentation',
            'OT': 'Object Tracking',
            'TLD': 'Traffic Light Detection'
        };
        return labels[type] || type;
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'SUCCESS': return '#4CAF50';
            case 'FAILED':
            case 'FAILED_RESOURCE_UNAVAILABLE': return '#f44336';
            case 'IN_EXEC': return '#FF9800';
            case 'CREATED': return '#2196F3';
            default: return '#9E9E9E';
        }
    };

    const getExecutionLevelIcon = (level) => {
        switch (level) {
            case 'DEVICE': return '📱';
            case 'EDGE': return '🌐';
            case 'CLOUD': return '☁️';
            default: return '❓';
        }
    };

    const formatLatency = (startTime, endTime) => {
        if (!startTime || !endTime) return 'N/A';
        const latency = endTime - startTime;
        return `${latency.toFixed(3)}s`;
    };

    const taskStats = getTasksByStatus();
    const typeDistribution = getTaskTypeDistribution();
    const successRate = taskStats.total > 0 ? (taskStats.completed / (taskStats.completed + taskStats.failed) * 100).toFixed(2) : 0;

    return (
        <div className="task-panel">
            <h3>📋 Task Management</h3>

            {/* Task Statistics Overview */}
            <div className="task-overview">
                <div className="stat-card total">
                    <div className="stat-number">{taskStats.total}</div>
                    <div className="stat-label">Total Tasks</div>
                </div>
                <div className="stat-card completed">
                    <div className="stat-number">{taskStats.completed}</div>
                    <div className="stat-label">Completed</div>
                </div>
                <div className="stat-card failed">
                    <div className="stat-number">{taskStats.failed}</div>
                    <div className="stat-label">Failed</div>
                </div>
                <div className="stat-card active">
                    <div className="stat-number">{taskStats.active}</div>
                    <div className="stat-label">Active</div>
                </div>
            </div>

            {/* Success Rate */}
            <div className="success-rate">
                <div className="success-rate-label">Success Rate</div>
                <div className="success-rate-bar">
                    <div
                        className="success-rate-fill"
                        style={{ width: `${successRate}%` }}
                    ></div>
                </div>
                <div className="success-rate-text">{successRate}%</div>
            </div>

            {/* Task Type Distribution */}
            {Object.keys(typeDistribution).length > 0 && (
                <div className="task-types">
                    <h4>📊 Task Types</h4>
                    <div className="type-distribution">
                        {Object.entries(typeDistribution).map(([type, count]) => (
                            <div key={type} className="type-item">
                                <div className="type-info">
                                    <span className="type-name">{getTaskTypeLabel(type)}</span>
                                    <span className="type-code">({type})</span>
                                </div>
                                <div className="type-count">{count}</div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Recent Tasks */}
            {taskStats.tasks.length > 0 && (
                <div className="recent-tasks">
                    <h4>🕒 Recent Tasks</h4>
                    <div className="tasks-list">
                        {taskStats.tasks.map((task) => (
                            <div key={task.id} className="task-item">
                                <div className="task-header">
                                    <div className="task-id">Task #{task.id}</div>
                                    <div className="task-type-badge">{task.type}</div>
                                    <div
                                        className="task-status"
                                        style={{ color: getStatusColor(task.status) }}
                                    >
                                        {task.status}
                                    </div>
                                </div>

                                <div className="task-details">
                                    <div className="task-detail">
                                        <span className="detail-label">Vehicle:</span>
                                        <span className="detail-value">#{task.vehicle_id}</span>
                                    </div>

                                    {task.chosen_execution_level && (
                                        <div className="task-detail">
                                            <span className="detail-label">Executed on:</span>
                                            <span className="detail-value">
                                                {getExecutionLevelIcon(task.chosen_execution_level)} {task.chosen_execution_level}
                                            </span>
                                        </div>
                                    )}

                                    {task.chosen_execution_platform && (
                                        <div className="task-detail">
                                            <span className="detail-label">Platform:</span>
                                            <span className="detail-value">{task.chosen_execution_platform}</span>
                                        </div>
                                    )}

                                    {task.chosen_execution_model && (
                                        <div className="task-detail">
                                            <span className="detail-label">Model:</span>
                                            <span className="detail-value">{task.chosen_execution_model}</span>
                                        </div>
                                    )}

                                    <div className="task-detail">
                                        <span className="detail-label">Latency:</span>
                                        <span className="detail-value">
                                            {formatLatency(task.execution_start_time, task.execution_end_time)}
                                        </span>
                                    </div>

                                    <div className="task-detail">
                                        <span className="detail-label">Accuracy Req:</span>
                                        <span className="detail-value">{(task.min_accuracy * 100).toFixed(0)}%</span>
                                    </div>

                                    <div className="task-detail">
                                        <span className="detail-label">Max Latency:</span>
                                        <span className="detail-value">{task.max_latency}s</span>
                                    </div>

                                    <div className="task-detail">
                                        <span className="detail-label">Data Size:</span>
                                        <span className="detail-value">{task.data_size.toFixed(2)} MB</span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* No Tasks Message */}
            {simulationStatus === 'idle' && (
                <div className="no-tasks-message">
                    <div className="no-tasks-icon">📋</div>
                    <div className="no-tasks-text">
                        <h4>No Tasks Yet</h4>
                        <p>Start a simulation to see real-time task execution and statistics</p>
                    </div>
                </div>
            )}

            {/* Loading Message */}
            {simulationStatus === 'starting' && (
                <div className="loading-message">
                    <div className="loading-icon">⏳</div>
                    <div className="loading-text">Starting simulation...</div>
                </div>
            )}
        </div>
    );
}

export default TaskPanel;