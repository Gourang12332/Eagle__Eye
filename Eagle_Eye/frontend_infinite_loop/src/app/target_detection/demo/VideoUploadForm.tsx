"use client";

import React, { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { DemoCCTVFootage } from "@/types/interfaces";
import { ImageKitProvider, IKUpload } from "imagekitio-next";

const publicKey = process.env.NEXT_PUBLIC_PUBLIC_KEY;
const urlEndpoint = process.env.NEXT_PUBLIC_URL_ENDPOINT;
const BACKEND_URL = process.env.NEXT_PUBLIC_COLLAB_PUBLIC_URL;

type UploadResponse = {
  url: string;
};

type UploadError = {
  message?: string;
  [key: string]: unknown;
};

export default function VideoUploadForm() {
  console.log("Backend url: ", BACKEND_URL);

  const [formData, setFormData] = useState<DemoCCTVFootage>({
    cctv_id: "",
    location_name: "",
    coordinates: { lat: 0, lng: 0 },
    time_uploaded_at: new Date(),
    video_url: "",
  });
  const [loading, setLoading] = useState(false);

  const authenticator = useCallback(async () => {
    try {
      const response = await fetch("/api/auth");
      if (!response.ok) throw new Error(`Auth failed: ${response.statusText}`);
      return await response.json();
    } catch (error) {
      console.error("Authentication failed:", error);
      return {};
    }
  }, []);

  const handleUploadSuccess = useCallback((res: UploadResponse) => {
    console.log("Upload success:", res);
    setFormData((prev) => ({ ...prev, video_url: res.url }));
  }, []);

  const handleUploadError = useCallback((err: UploadError) => {
    console.error("Upload error:", err);
  }, []);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const { name, value } = e.target;
      setFormData((prev) => {
        if (name === "lat" || name === "lng") {
          return {
            ...prev,
            coordinates: {
              ...prev.coordinates,
              [name]: value ? parseFloat(value) : "",
            },
          };
        }
        return { ...prev, [name]: value };
      });
    },
    []
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.video_url) {
      console.error("No video uploaded.");
      return;
    }

    setLoading(true);
    console.log("Sending request to backend:", formData);
    console.log("Backend url: ", BACKEND_URL);

    const requestBody = {
      CCTV_id: String(formData.cctv_id),
      location: formData.location_name,
      latlong: [formData.coordinates.lat, formData.coordinates.lng],
      timestamp: new Date().toISOString(),
      Url: formData.video_url,
    };

    try {
      const response = await fetch(`${BACKEND_URL}/cctv-data`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      console.log("Request sent, waiting for response...");

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      console.log("Response received from backend:", data);

      window.location.reload();
    } catch (error) {
      console.error("Upload error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Add New CCTV Video</CardTitle>
        <CardDescription>Enter the CCTV footage details</CardDescription>
      </CardHeader>

      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="cctv_id">CCTV ID</Label>
            <Input
              id="cctv_id"
              name="cctv_id"
              placeholder="Enter CCTV ID"
              value={formData.cctv_id}
              onChange={handleInputChange}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="location_name">Location Name</Label>
            <Input
              id="location_name"
              name="location_name"
              placeholder="Enter location name"
              value={formData.location_name}
              onChange={handleInputChange}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="lat">Latitude</Label>
              <Input
                id="lat"
                name="lat"
                type="number"
                step="any"
                placeholder="Enter latitude"
                value={formData.coordinates.lat || ""}
                onChange={handleInputChange}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="lng">Longitude</Label>
              <Input
                id="lng"
                name="lng"
                type="number"
                step="any"
                placeholder="Enter longitude"
                value={formData.coordinates.lng || ""}
                onChange={handleInputChange}
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="video">Upload Video</Label>
            <ImageKitProvider
              publicKey={publicKey}
              urlEndpoint={urlEndpoint}
              authenticator={authenticator}
            >
              <IKUpload
                className="bg-neutral-950 rounded-lg text-white px-4 py-2 text-sm"
                onError={handleUploadError}
                onSuccess={handleUploadSuccess}
              />
            </ImageKitProvider>
          </div>

          {formData.video_url && (
            <div className="mt-4">
              <Label>Video Preview:</Label>
              <video
                src={formData.video_url}
                controls
                className="w-full rounded-lg shadow-lg"
              />
            </div>
          )}
        </CardContent>

        <CardFooter>
          <Button type="submit" disabled={loading}>
            {loading ? "Submitting..." : "Submit"}
          </Button>
        </CardFooter>
      </form>
    </Card>
  );
}
