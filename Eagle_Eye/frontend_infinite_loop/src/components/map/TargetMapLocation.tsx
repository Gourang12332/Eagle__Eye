"use client";

import React, { useEffect, useState } from "react";
import {
  APIProvider,
  Map,
  AdvancedMarker,
  InfoWindow,
  useMap,
} from "@vis.gl/react-google-maps";


type Coordinates = {
  lat: number;
  lng: number;
};

type TargetFoundAt = {
  timestamp: number;
  coordinates: Coordinates;
  cctv_id: string;
  location_name: string;
};

type TargetMapLocationProps = {
  target_found_at: TargetFoundAt[];
};


const MARKER_CHANGE_TIME_1 = 6 * 60 * 60 * 1000;
const MARKER_CHANGE_TIME_2 = 24 * 60 * 60 * 1000;
const MARKER_CHANGE_TIME_3 = 7 * 24 * 60 * 60 * 1000;

const getMarkerSize = (timestamp: number) => {
  const now = Date.now();
  const diff = now - timestamp;

  if (diff <= MARKER_CHANGE_TIME_1) return 36;
  if (diff <= MARKER_CHANGE_TIME_2) return 30;
  if (diff <= MARKER_CHANGE_TIME_3) return 24;
  return 20;
};

const getMarkerColor = (timestamp: number) => {
  const now = Date.now();
  const diff = now - timestamp;

  if (diff <= MARKER_CHANGE_TIME_1)
    return { background: "#FF3B30", border: "#B71C1C" };
  if (diff <= MARKER_CHANGE_TIME_2)
    return { background: "#FF9800", border: "#E65100" };
  if (diff <= MARKER_CHANGE_TIME_3)
    return { background: "#FFD600", border: "#FFAB00" };
  return { background: "#9E9E9E", border: "#616161" };
};


const TargetMarkers = React.memo(({ target_found_at }: TargetMapLocationProps) => {
  const [selectedMarker, setSelectedMarker] = useState<TargetFoundAt | null>(null);

  const sortedLocations = [...target_found_at].sort(
    (a, b) => a.timestamp - b.timestamp
  );

  return (
    <>
      {sortedLocations.map((geolocation, index) => {
        const { background, border } = getMarkerColor(geolocation.timestamp);
        const size = getMarkerSize(geolocation.timestamp);
        const markerNumber = index + 1;

        return (
          <AdvancedMarker
            position={geolocation.coordinates}
            onClick={() => setSelectedMarker(geolocation)}
            key={`${geolocation.cctv_id}-${geolocation.timestamp}`}
          >
            <div className="relative flex items-center justify-center">
              <div
                className="absolute w-full h-full rounded-full opacity-50 animate-ping"
                style={{
                  width: `${size * 1.6}px`,
                  height: `${size * 1.6}px`,
                  backgroundColor: background,
                }}
              />
              <div
                className="relative flex items-center justify-center text-white font-bold border-2 rounded-full"
                style={{
                  width: `${size}px`,
                  height: `${size}px`,
                  backgroundColor: background,
                  borderColor: border,
                  fontSize: "14px",
                }}
              >
                {markerNumber}
              </div>
            </div>
          </AdvancedMarker>
        );
      })}

      {selectedMarker && (
        <InfoWindow
          position={selectedMarker.coordinates}
          onCloseClick={() => setSelectedMarker(null)}
        >
          <div className="p-2 text-sm font-medium">
            <p className="font-bold">{selectedMarker.location_name}</p>
            <p className="text-gray-600">
              <strong>Time:</strong>{" "}
              {new Intl.DateTimeFormat("en-GB", {
                day: "2-digit",
                month: "2-digit",
                year: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              }).format(new Date(selectedMarker.timestamp))}
            </p>
            <p className="text-gray-600">
              <strong>CCTV ID:</strong> {selectedMarker.cctv_id}
            </p>
          </div>
        </InfoWindow>
      )}
    </>
  );
});

// Component: FitMapToBounds
const FitMapToBounds = React.memo(({ target_found_at }: TargetMapLocationProps) => {
  const map = useMap();

  useEffect(() => {
    if (!map || target_found_at.length === 0 || !window.google?.maps) return;

    const bounds = new window.google.maps.LatLngBounds();
    target_found_at.forEach(({ coordinates }) => bounds.extend(coordinates));

    map.fitBounds(bounds, 300);
  }, [map, target_found_at]);

  return null;
});

// Main Map Component
export default function TargetMapLocation({ target_found_at }: TargetMapLocationProps) {
  const apiKey = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY;
  const mapId = process.env.NEXT_PUBLIC_GOOGLE_MAPS_MAP_ID;

  if (!apiKey) {
    console.error("Google Maps API key is missing");
    return <p>Google Maps API key is required</p>;
  }

  return (
    <APIProvider apiKey={apiKey}>
      <Map
        style={{ flex: 1, height: "100vh", zIndex: 1 }}
        defaultCenter={{ lat: 20.5937, lng: 78.9629 }}
        defaultZoom={5.5}
        gestureHandling="greedy"
        disableDefaultUI={true}
        mapId={mapId}
        reuseMaps={true}
      >
        <TargetMarkers target_found_at={target_found_at} />
        <FitMapToBounds target_found_at={target_found_at} />
      </Map>
    </APIProvider>
  );
}
