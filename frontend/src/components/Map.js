// src/components/Map.js
import React, { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, Polyline } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for default markers in react-leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom vehicle icon
// const vehicleIcon = new L.DivIcon({
//     html: `< div style = "
// background - color: #4285f4;
// border - radius: 50 %;
// width: 25px;
// height: 25px;
// border: 2px solid white;
// box - shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
// "></div>`,
//     iconSize: [25, 25],
//     iconAnchor: [12.5, 12.5],
//     className: 'vehicle-marker'
// });
// Custom edge server icon
const vehiclePositionIcon = new L.DivIcon({
    html: `<div style="
    background-color: #4285f4; 
    border-radius: 50%; 
    width: 15px; 
    height: 15px; 
    border: 2px solid white; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.4);
  "></div>`,
    iconSize: [15, 15],
    iconAnchor: [7.5, 7.5],
    className: 'edge-marker'
});
const vehicleDestinationIcon = new L.DivIcon({
    html: `<div style="
    background-color: #24ae46ff; 
    border-radius: 50%; 
    width: 15px; 
    height: 15px; 
    border: 2px solid white; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.4);
  "></div>`,
    iconSize: [15, 15],
    iconAnchor: [7.5, 7.5],
    className: 'edge-marker'
});

// Custom edge server icon
const edgeIcon = new L.DivIcon({
    html: `<div style="
    background-color: #ea4335; 
    border-radius: 50%; 
    width: 25px; 
    height: 25px; 
    border: 3px solid white; 
    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
  "></div>`,
    iconSize: [25, 25],
    iconAnchor: [12.5, 12.5],
    className: 'edge-marker'
});

const Map = ({ simulationData, currentConfig }) => {
    // console.log('Map component rendered with:', { simulationData, currentConfig });

    // Default center coordinates based on city
    const getCityCenter = () => {
        if (!currentConfig || !currentConfig.city) {
            return [48.8566, 2.3522]; // Default to Paris
        }

        const cityCoords = {
            paris: [48.8566, 2.3522],
            berlin: [52.5200, 13.4050],
            algiers: [36.7372, 3.0869],
            alger: [36.7578, 3.05775],
            lille: [50.631583072533594, 3.057713469569928],
            bordeaux: [44.83911345093876, -0.5775176280843897],
            shangai: [31.230416666666667, 121.47222222222221],
            los_angeles: [34.05222222222222, -118.24277777777777],
            tokyo: [35.6894875, 139.6917064],
            rennes: [48.110825, -1.678152],
            lyon: [45.764043, 4.835658],
            london: [51.509865, -0.118092],
            new_york: [40.7127837, -74.0059413],
            moscow: [55.755826, 37.617298],
            rome: [41.9027835, 12.4963655],
            kigali: [-1.9484010053997387, 30.0953033638107],
            marrakech: [31.62949, -7.984659]
        };

        return cityCoords[currentConfig.city] || [48.8566, 2.3522];
    };

    const center = getCityCenter();
    const zoom = 13;

    // If no simulation data, show loading state
    if (!simulationData) {
        return (
            <div style={{
                height: '100%',
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: '#f0f0f0',
                color: '#666'
            }}>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '24px', marginBottom: '10px' }}>🗺️</div>
                    <div>Map loading...</div>
                    <div style={{ fontSize: '12px', marginTop: '5px' }}>
                        Waiting for simulation data
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div style={{ height: '100%', width: '100%' }}>
            <MapContainer
                center={center}
                zoom={zoom}
                style={{ height: '100%', width: '100%' }}
                zoomControl={true}
            >
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                {/* Render edge servers with coverage areas */}
                {simulationData.edge_servers && simulationData.edge_servers.map((edge) => (
                    <React.Fragment key={`edge-${edge.id}`}>
                        {/* Coverage area circle - Ensure coverage_radius is used, fallback to 2km if missing */}
                        <Circle
                            center={edge.position || center}
                            radius={(edge.coverage_radius || 2.0) * 1000} // Convert km to meters, fallback to 2km
                            pathOptions={{
                                color: '#ea4335',
                                fillColor: '#ea4335',
                                fillOpacity: 0.2, // Increased opacity for visibility
                                weight: 2,
                                dashArray: '5, 5'
                            }}
                        />
                        {/* Edge server marker */}
                        <Marker position={edge.position || center} icon={edgeIcon}>
                            <Popup>
                                <div style={{ minWidth: '200px' }}>
                                    <h3 style={{ margin: '0 0 10px 0', color: '#ea4335' }}>
                                        {edge.name || `Edge Server ${edge.id}`}
                                    </h3>
                                    <p><strong>Position:</strong> [{(edge.position || center)[0].toFixed(4)}, {(edge.position || center)[1].toFixed(4)}]</p>
                                    <p><strong>Coverage:</strong> {(edge.coverage_radius || 2.0)} km</p>
                                    <p><strong>Queue:</strong> {edge.queue_size || 0}/{edge.queue_capacity || 0}</p>
                                    <p><strong>Active Tasks:</strong> {edge.active_tasks || 0}</p>
                                    <p><strong>Total Processed:</strong> {edge.total_processed || 0}</p>
                                </div>
                            </Popup>
                        </Marker>
                    </React.Fragment>
                ))}

                {/* Render vehicles with trajectories and updated positions */}
                {simulationData.vehicles && simulationData.vehicles.map((vehicle) => {
                    const currentPosition = vehicle.trajectory.current_position || center; // Fallback to center if no position
                    const futurePosition = vehicle.trajectory.destination_location || center; // Fallback to center if no position
                    const traversedPath = vehicle.trajectory.traveled || []; // Array of [lat, lng] points
                    const plannedPath = vehicle.trajectory.planned || []; // Array of [lat, lng] points
                    return (
                        <React.Fragment key={`vehicle-${vehicle.id}`}>
                            {/* Traversed trajectory (grey solid line) */}
                            {traversedPath.length > 1 && (
                                <Polyline
                                    positions={traversedPath}
                                    pathOptions={{
                                        color: 'grey',
                                        weight: 3,
                                        opacity: 0.8
                                    }}
                                />
                            )}
                            {/* Planned trajectory (blue dashed line) */}
                            {plannedPath.length > 1 && (
                                <Polyline
                                    positions={plannedPath}
                                    pathOptions={{
                                        color: 'blue',
                                        weight: 2,
                                        // dashArray: '5, 5',
                                        opacity: 1
                                    }}
                                />
                            )}
                            {/* Vehicle marker at current position */}
                            <Marker position={currentPosition} icon={vehiclePositionIcon}>
                                <Popup>
                                    <div style={{ minWidth: '200px' }}>
                                        <h3 style={{ margin: '0 0 10px 0', color: '#4285f4' }}>
                                            {vehicle.name || `Vehicle ${vehicle.id}`}
                                        </h3>
                                        <p><strong>Position:</strong> [{currentPosition[0].toFixed(4)}, {currentPosition[1].toFixed(4)}]</p>
                                        <p><strong>Position:</strong> [{futurePosition[0].toFixed(4)}, {futurePosition[1].toFixed(4)}]</p>
                                        <p><strong>Total Tasks:</strong> {vehicle.total_tasks || 0}</p>
                                        <p><strong>Completed:</strong> {vehicle.tasks_completed || 0}</p>
                                        <p><strong>Active Tasks:</strong> {vehicle.current_task_count || 0}</p>
                                        <p><strong>Current Edge:</strong> Edge-{vehicle.current_edge_server || 'N/A'}</p>
                                        <p><strong>Trip Status:</strong>
                                            <span style={{ color: vehicle.trip_finished ? '#34a853' : '#fbbc04' }}>
                                                {vehicle.trip_finished ? 'Completed' : 'In Progress'}
                                            </span>
                                        </p>
                                    </div>
                                </Popup>
                                <p>{vehicle.id}</p>
                            </Marker>
                            {/* Vehicle marker at destination position */}
                            <Marker position={futurePosition} icon={vehicleDestinationIcon}>
                                <Popup>
                                    <div style={{ minWidth: '200px' }}>
                                        <h3 style={{ margin: '0 0 10px 0', color: '#4285f4' }}>
                                            {vehicle.name || `Vehicle ${vehicle.id}`}
                                        </h3>
                                        <p><strong>Position:</strong> [{futurePosition[0].toFixed(4)}, {futurePosition[1].toFixed(4)}]</p>
                                        <p><strong>Position:</strong> [{futurePosition[0].toFixed(4)}, {futurePosition[1].toFixed(4)}]</p>
                                        <p><strong>Total Tasks:</strong> {vehicle.total_tasks || 0}</p>
                                        <p><strong>Completed:</strong> {vehicle.tasks_completed || 0}</p>
                                        <p><strong>Active Tasks:</strong> {vehicle.current_task_count || 0}</p>
                                        <p><strong>Current Edge:</strong> Edge-{vehicle.current_edge_server || 'N/A'}</p>
                                        <p><strong>Trip Status:</strong>
                                            <span style={{ color: vehicle.trip_finished ? '#34a853' : '#fbbc04' }}>
                                                {vehicle.trip_finished ? 'Completed' : 'In Progress'}
                                            </span>
                                        </p>
                                    </div>
                                </Popup>
                                <p>{vehicle.id}</p>
                            </Marker>
                        </React.Fragment>
                    );
                })}
            </MapContainer>
        </div>
    );
};

export default Map;
