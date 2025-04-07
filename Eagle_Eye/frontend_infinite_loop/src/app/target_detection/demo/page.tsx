"use client";

import { useState } from "react";
import { SidebarProvider } from "@/components/ui/sidebar";
import VideoSidebar from "./VideoSidebar";
import VideoUploadForm from "./VideoUploadForm";

export default function VideoManagementPage() {
  const [videos] = useState([
    { id: "vid001", name: "Entrance Camera - Main Building" },
    { id: "vid002", name: "Parking Lot - North Side" },
    { id: "vid003", name: "Hallway - Floor 2" },
    { id: "vid004", name: "Reception Area" },
    { id: "vid005", name: "Loading Dock" },
    { id: "vid006", name: "Emergency Exit - East Wing" },
    { id: "vid007", name: "Server Room" },
    { id: "vid008", name: "Cafeteria" },
  ]);

  return (
    <SidebarProvider>
      <div className="flex min-h-screen w-full bg-background">
        <VideoSidebar videos={videos} />
        <main className="flex-1 p-6">
          <div className="mx-auto max-w-3xl flex flex-col">
            <h1 className="mb-6 text-2xl font-bold">Video Management</h1>
            <VideoUploadForm />
          </div>
        </main>
      </div>
    </SidebarProvider>
  );
}
