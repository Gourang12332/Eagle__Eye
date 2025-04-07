"use client";

import { Video, Plus } from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarFooter,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupContent,
} from "@/components/ui/sidebar";

interface VideoSidebarProps {
  videos: { id: string; name: string }[];
}

export default function VideoSidebar({ videos }: VideoSidebarProps) {
  return (
    <Sidebar className="border-r">
      <SidebarHeader className="border-b">
        <div className="flex items-center gap-2 px-4 py-2">
          <Video className="h-5 w-5" />
          <h2 className="text-lg font-semibold">CCTV Videos</h2>
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Uploaded Videos</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {videos.map((video) => (
                <SidebarMenuItem key={video.id}>
                  <SidebarMenuButton>
                    <Video className="h-4 w-4" />
                    <span>{video.name}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton>
              <Plus className="h-4 w-4" />
              <span>Add New Video</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  );
}
