import { TargetLocation } from "@/types/interfaces";

const now = Date.now();

const example_target_found_at_ayush: TargetLocation[] = [
  {
    cctv_id: "CAM-001",
    location_name: "City Mall",
    coordinates: { lat: 25.1825, lng: 75.8511 },
    timestamp: new Date(now - 5 * 60 * 60 * 1000), // ~5 hours ago
    confidence: 0.74,
  },
  {
    cctv_id: "CAM-002",
    location_name: "Aerodrome Circle",
    coordinates: { lat: 25.1642, lng: 75.8519 },
    timestamp: new Date(now - 4 * 60 * 60 * 1000), // ~4 hours ago
    confidence: 0.79,
  },
];

const example_target_found_at_gourang: TargetLocation[] = [
  {
    cctv_id: "CAM-003",
    location_name: "Kota Barrage",
    coordinates: { lat: 25.1652, lng: 75.8577 },
    timestamp: new Date(now - 12 * 60 * 60 * 1000), // ~12 hours ago
    confidence: 0.85,
  },
  {
    cctv_id: "CAM-004",
    location_name: "Allen Samyak",
    coordinates: { lat: 25.1638, lng: 75.8425 },
    timestamp: new Date(now - 3 * 24 * 60 * 60 * 1000), // ~3 days ago
    confidence: 0.88,
  },
];

const example_target_found_at_ashmit: TargetLocation[] = [
  {
    cctv_id: "CAM-005",
    location_name: "Indira Market",
    coordinates: { lat: 25.1753, lng: 75.8615 },
    timestamp: new Date(now - 6 * 24 * 60 * 60 *1000), // ~6 days ago
    confidence: 0.92,
  },
  {
    cctv_id: "CAM-006",
    location_name: "Seven Wonders Park",
    coordinates: { lat: 25.1694, lng: 75.8356 },
    timestamp: new Date(now - 10 * 24 * 60 * 60 * 1000), // ~10 days ago
    confidence: 0.83,
  },
];

export {
  example_target_found_at_ayush,
  example_target_found_at_gourang,
  example_target_found_at_ashmit,
};
