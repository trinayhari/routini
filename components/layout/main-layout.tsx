"use client";

import React, { useState } from "react";
import { Navbar } from "@/components/layout/navbar";
import { Sidebar } from "@/components/sidebar/sidebar";
import { RoutingStrategy, ChatMessage } from "@/types";
import { Toaster } from "@/components/ui/toaster";

interface MainLayoutProps {
  children: React.ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  const [strategy, setStrategy] = useState<RoutingStrategy>("balanced");
  const [developerMode, setDeveloperMode] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  // Function to update messages from ChatInterface
  const handleMessagesUpdate = (newMessages: ChatMessage[]) => {
    setMessages(newMessages);
  };

  return (
    <div className="flex flex-col h-screen">
      <Navbar />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          strategy={strategy}
          developerMode={developerMode}
          onStrategyChange={setStrategy}
          onDeveloperModeChange={setDeveloperMode}
          messages={messages}
        />
        <main className="flex-1 overflow-auto">
          {/* Clone children and pass props */}
          {React.Children.map(children, (child) => {
            if (React.isValidElement(child)) {
              return React.cloneElement(child as React.ReactElement<any>, {
                strategy,
                developerMode,
                onMessagesUpdate: handleMessagesUpdate,
              });
            }
            return child;
          })}
        </main>
      </div>
      <Toaster />
    </div>
  );
}
