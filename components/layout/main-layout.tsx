"use client";

import React, { useState } from "react";
import { Navbar } from "@/components/layout/navbar";
import { Sidebar } from "@/components/sidebar/sidebar";
import { RoutingStrategy, ChatMessage } from "@/types";
import { Toaster } from "@/components/ui/toaster";
import { MODELS } from "@/lib/mock-data";

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
          modelCosts={MODELS}
        />
        <main className="flex-1 overflow-auto">
          {/* Clone children and pass props */}
          {React.Children.map(children, (child) => {
            if (React.isValidElement(child)) {
              // Ensure all expected props are passed or handled if optional
              const childProps: any = {
                strategy,
                developerMode,
                onMessagesUpdate: handleMessagesUpdate,
              };
              // If the child is ComparePage, we don't want to pass onMessagesUpdate
              // This check is a bit brittle; ideally, context or more specific prop drilling is used.
              if (child.type && (child.type as any).name === "ComparePage") {
                delete childProps.onMessagesUpdate;
              }

              return React.cloneElement(
                child as React.ReactElement<any>,
                childProps
              );
            }
            return child;
          })}
        </main>
      </div>
      <Toaster />
    </div>
  );
}
