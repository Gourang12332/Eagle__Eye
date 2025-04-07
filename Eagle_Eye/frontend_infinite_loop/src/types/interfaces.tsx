export interface TargetLocation {
  cctv_id: string;
  location_name: string;
  coordinates: google.maps.LatLngLiteral;
  timestamp: Date;
  confidence?: number;
}

export interface DemoCCTVFootage {
  cctv_id: string;
  location_name: string;
  coordinates: google.maps.LatLngLiteral;
  time_uploaded_at: Date;
  video_url: string;
}
