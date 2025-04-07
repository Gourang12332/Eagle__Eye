"use client";

import LocationCard from "./LocationCard";
import { TargetLocation } from "@/types/interfaces";

export default function LocationList({
  target_found_at,
}: {
  target_found_at: TargetLocation[];
}) {
  const sortedLocations = [...target_found_at].sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
  );

  return (
    <section className="w-full h-full max-w-sm p-4 absolute top-0 right-0 z-10 flex flex-col gap-2 bg-transparent rounded-lg">
      <div className="px-4 py-3 border-b bg-white rounded-lg">
        <h3 className="text-lg font-semibold">Recent Detections</h3>
      </div>

      <div className="overflow-y-auto max-h-screen flex flex-col gap-2">
        {sortedLocations.map((detection) => (
          <LocationCard
            key={detection.cctv_id + detection.timestamp}
            location_name={detection.location_name}
            cctv_id={detection.cctv_id}
            coordinates={detection.coordinates}
            timestamp={detection.timestamp}
            confidence={detection.confidence}
          />
        ))}
      </div>
    </section>
  );
}
