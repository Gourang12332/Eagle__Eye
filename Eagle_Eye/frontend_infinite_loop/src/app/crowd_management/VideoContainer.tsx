import { useState } from "react";
import { ImageKitProvider, IKVideo } from "imagekitio-next";
import { Loader2 } from "lucide-react";

type VideoContainerProps = {
  videoURL: string | null;
};

export default function VideoContainer({ videoURL }:VideoContainerProps) {
  const [loading, setLoading] = useState(false);

  return (
    <div className="flex items-center justify-center min-h-64 p-4">
      {videoURL ? (
        <ImageKitProvider urlEndpoint={process.env.NEXT_PUBLIC_URL_ENDPOINT}>
          <div className="relative">
            {loading && (
              <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded-lg">
                <Loader2 className="w-10 h-10 animate-spin text-white" />
              </div>
            )}
            <IKVideo
              src={videoURL}
              className="max-h-[50vh] object-cover"
              controls={true}
              onLoadStart={() => setLoading(true)}
              onLoadedData={() => setLoading(false)}
            />
          </div>
        </ImageKitProvider>
      ) : (
        <div className="flex flex-col items-center justify-center p-8 text-gray-600 dark:text-gray-300">
          <svg
            className="w-16 h-16 text-gray-400 dark:text-gray-500"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15 10l4.553 2.276A1 1 0 0120 13v2a1 1 0 01-.447.832L15 18m-6 0l-4.553-2.276A1 1 0 014 15v-2a1 1 0 01.447-.832L9 10m6 0V6a2 2 0 00-2-2h-2a2 2 0 00-2 2v4m6 0H9"
            ></path>
          </svg>
          <p className="mt-4 text-center text-sm">
            No video uploaded. Please upload a video to preview.
          </p>
        </div>
      )}
    </div>
  );
}
