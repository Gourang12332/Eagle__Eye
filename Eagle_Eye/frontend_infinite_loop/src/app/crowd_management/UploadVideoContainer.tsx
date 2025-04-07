"use client";

import React, { useState } from "react";
import { ImageKitProvider, IKUpload } from "imagekitio-next";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";

const publicKey = process.env.NEXT_PUBLIC_PUBLIC_KEY!;
const urlEndpoint = process.env.NEXT_PUBLIC_URL_ENDPOINT!;

type UploadVideoContainerProps = {
  setVideoURL: (url: string) => void;
};

// ImageKit success and error response types (basic)
type ImageKitUploadSuccess = {
  fileId: string;
  name: string;
  url: string;
  filePath: string;
  size: number;
  height?: number;
  width?: number;
  thumbnailUrl?: string;
};

type ImageKitUploadError = {
  message?: string;
  statusCode?: number;
  response?: unknown;
};

export default function UploadVideoContainer({
  setVideoURL,
}: UploadVideoContainerProps) {
  const [loading, setLoading] = useState(false);

  const authenticator = async () => {
    try {
      const response = await fetch("/api/auth");

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `Request failed with status ${response.status}: ${errorText}`
        );
      }

      const { signature, expire, token } = await response.json();
      return { signature, expire, token };
    } catch (error) {
      console.error("Authentication request failed:", error);
      throw error;
    }
  };

  const onError = (err: ImageKitUploadError) => {
    console.error("Upload Error:", err);
    setLoading(false);
  };

  const onSuccess = (res: ImageKitUploadSuccess) => {
    console.log("Upload Success:", res);
    setVideoURL(`${urlEndpoint}${res.filePath}`);
    setLoading(false);
  };

  return (
    <ImageKitProvider
      publicKey={publicKey}
      urlEndpoint={urlEndpoint}
      authenticator={authenticator}
    >
      <IKUpload
        onError={onError}
        onSuccess={onSuccess}
        className="hidden"
        id="file-upload"
        onUploadStart={() => setLoading(true)}
      />
      <label htmlFor="file-upload">
        <Button asChild disabled={loading}>
          <span className="flex items-center gap-2">
            {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : null}
            {loading ? "Uploading..." : "Upload Video"}
          </span>
        </Button>
      </label>
    </ImageKitProvider>
  );
}
