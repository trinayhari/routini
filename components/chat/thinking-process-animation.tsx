"use client";

import { useState, useEffect } from "react";
import { Loader2 } from "lucide-react";

const GENERIC_THINKING_MESSAGES = [
  "Analyzing your prompt...",
  "Considering model capabilities...",
  "Evaluating complexity and context...",
  "Balancing cost, speed, and quality...",
  "Cross-referencing routing rules...",
];

interface ThinkingProcessAnimationProps {
  phase: "thinking" | "revealing";
  rationale?: string;
  model?: string;
}

export function ThinkingProcessAnimation({
  phase,
  rationale,
  model,
}: ThinkingProcessAnimationProps) {
  const [currentMessageIndex, setCurrentMessageIndex] = useState(0);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (phase === "thinking") {
      interval = setInterval(() => {
        setCurrentMessageIndex(
          (prevIndex) => (prevIndex + 1) % GENERIC_THINKING_MESSAGES.length
        );
      }, 2000); // Change message every 2 seconds
    }
    return () => clearInterval(interval);
  }, [phase]);

  let content;
  if (phase === "revealing") {
    content = (
      <div className="space-y-1">
        <p className="animate-text-fade-in">
          {rationale
            ? `Finalizing selection based on: ${rationale}`
            : "Finalizing model selection..."}
        </p>
        {model && (
          <p className="animate-text-fade-in font-semibold">
            Chosen Model: {model}
          </p>
        )}
      </div>
    );
  } else {
    content = (
      <p className="animate-text-fade-in">
        {GENERIC_THINKING_MESSAGES[currentMessageIndex]}
      </p>
    );
  }

  return (
    <div className="flex items-start space-x-2 p-3 bg-muted/50 rounded-lg shadow-sm max-w-[70%]">
      <Loader2 className="h-5 w-5 animate-spin text-primary mt-0.5" />
      <div className="text-sm text-muted-foreground">{content}</div>
    </div>
  );
}
