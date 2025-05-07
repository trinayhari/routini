"use client";

import { useState, useRef, useEffect } from "react";
import { RoutingStrategy, ChatMessage } from "@/types";
import { Message } from "@/components/chat/message";
import { MessageInput } from "@/components/chat/message-input";
import { generateResponse } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";

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

    // Add user message
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Send request to API
      const response = await generateResponse({
        prompt: content,
        strategy,
        previousMessages: messages,
      });

      // Add assistant message with response
      setMessages((prev) => [...prev, response.message]);
    } catch (error) {
      console.error("Error sending message:", error);

      // Add error message
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
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
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

        {isLoading && (
          <div className="flex items-start space-x-2 animate-in fade-in-0 slide-in-from-bottom-3">
            <div className="space-y-2 max-w-[70%]">
              <Skeleton className="h-12 w-[250px] rounded-lg" />
              <Skeleton className="h-4 w-[100px]" />
            </div>
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
