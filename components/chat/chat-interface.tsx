"use client";

import { useState, useRef, useEffect } from "react";
import { RoutingStrategy, ChatMessage, ResponseMetadata } from "@/types";
import { Message } from "@/components/chat/message";
import { MessageInput } from "@/components/chat/message-input";
import { generateResponse } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { ThinkingProcessAnimation } from "@/components/chat/thinking-process-animation";

interface ChatInterfaceProps {
  strategy: RoutingStrategy;
  developerMode: boolean;
  onMessagesUpdate?: (messages: ChatMessage[]) => void;
}

export function ChatInterface({
  strategy,
  developerMode,
  onMessagesUpdate,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [thinkingPhase, setThinkingPhase] = useState<
    "thinking" | "revealing" | "idle"
  >("idle");
  const [currentRationale, setCurrentRationale] = useState<string | undefined>(
    undefined
  );
  const [currentModel, setCurrentModel] = useState<string | undefined>(
    undefined
  );
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Update parent component when messages change
  useEffect(() => {
    onMessagesUpdate?.(messages);
  }, [messages, onMessagesUpdate]);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);

    setIsLoading(true);
    setThinkingPhase("thinking");
    setCurrentRationale(undefined);
    setCurrentModel(undefined);

    try {
      const response = await generateResponse({
        prompt: content,
        strategy,
        previousMessages: messages,
      });

      // Revealing phase
      setCurrentRationale(response.message.metadata?.routingRationale);
      setCurrentModel(response.message.metadata?.model);
      setThinkingPhase("revealing");

      // Wait for a bit to show the revealing animation
      await new Promise((resolve) => setTimeout(resolve, 3000));

      setMessages((prev) => [...prev, response.message]);
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage: ChatMessage = {
        id: Date.now().toString() + "-error",
        role: "assistant",
        content:
          "Sorry, there was an error processing your request. Please try again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setThinkingPhase("idle");
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && thinkingPhase === "idle" ? (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            No messages yet. Start a conversation!
          </div>
        ) : (
          messages.map((message) => (
            <Message
              key={message.id}
              message={message}
              showTrace={developerMode}
            />
          ))
        )}

        {isLoading && thinkingPhase !== "idle" && (
          <div className="flex items-start space-x-2 animate-in fade-in-0 slide-in-from-bottom-3">
            <ThinkingProcessAnimation
              phase={thinkingPhase}
              rationale={currentRationale}
              model={currentModel}
            />
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="border-t p-4">
        <MessageInput onSendMessage={handleSendMessage} isLoading={isLoading} />
      </div>
    </div>
  );
}
