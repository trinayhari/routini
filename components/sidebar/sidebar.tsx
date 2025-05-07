"use client";

import { useState } from "react";
import { RoutingStrategy, ChatMessage } from "@/types";
import { StrategySelector } from "@/components/sidebar/strategy-selector";
import { DeveloperModeToggle } from "@/components/sidebar/developer-mode-toggle";
import { ChatSummary } from "@/components/chat/chat-summary";
import { Separator } from "@/components/ui/separator";

interface SidebarProps {
  strategy: RoutingStrategy;
  developerMode: boolean;
  onStrategyChange: (strategy: RoutingStrategy) => void;
  onDeveloperModeChange: (enabled: boolean) => void;
  messages?: ChatMessage[];
}

export function Sidebar({
  strategy,
  developerMode,
  onStrategyChange,
  onDeveloperModeChange,
  messages = [],
}: SidebarProps) {
  return (
    <aside className="w-full md:w-64 h-full border-r bg-background p-4 space-y-4">
      <div className="text-sm font-medium">Router Settings</div>
      <div className="space-y-4">
        <StrategySelector value={strategy} onChange={onStrategyChange} />
        <Separator />
        <DeveloperModeToggle
          value={developerMode}
          onChange={onDeveloperModeChange}
        />
        <Separator />
        <ChatSummary messages={messages} />
      </div>
    </aside>
  );
}
