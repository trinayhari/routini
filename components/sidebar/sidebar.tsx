"use client";

import { useState, useEffect } from "react";
import { RoutingStrategy, ChatMessage, ResponseMetadata } from "@/types";
import { StrategySelector } from "@/components/sidebar/strategy-selector";
import { DeveloperModeToggle } from "@/components/sidebar/developer-mode-toggle";
import { ChatSummary } from "@/components/chat/chat-summary";
import { Separator } from "@/components/ui/separator";

export interface ModelCost {
  name: string;
  costPerToken: number;
  qualityScore: number;
}

export interface ModelCosts {
  [key: string]: ModelCost;
}

interface SidebarProps {
  strategy: RoutingStrategy;
  developerMode: boolean;
  onStrategyChange: (strategy: RoutingStrategy) => void;
  onDeveloperModeChange: (enabled: boolean) => void;
  messages?: ChatMessage[];
  modelCosts?: ModelCosts;
}

const GPT4_MODEL_ID = "openai/gpt-4";

export function Sidebar({
  strategy,
  developerMode,
  onStrategyChange,
  onDeveloperModeChange,
  messages = [],
  modelCosts = {},
}: SidebarProps) {
  const [totalSavings, setTotalSavings] = useState(0);
  const [actualCost, setActualCost] = useState(0);
  const [hypotheticalCost, setHypotheticalCost] = useState(0);
  const [savingsPercentage, setSavingsPercentage] = useState(0);

  useEffect(() => {
    if (
      !messages ||
      messages.length === 0 ||
      !modelCosts ||
      Object.keys(modelCosts).length === 0
    ) {
      setTotalSavings(0);
      setActualCost(0);
      setHypotheticalCost(0);
      setSavingsPercentage(0);
      return;
    }

    let currentActualCost = 0;
    let currentHypotheticalCost = 0;
    const gpt4CostPerToken = modelCosts[GPT4_MODEL_ID]?.costPerToken;

    if (typeof gpt4CostPerToken !== "number") {
      console.warn(
        `Cost for GPT-4 model ('${GPT4_MODEL_ID}') not found or invalid in modelCosts. Savings cannot be calculated accurately.`
      );
    }

    messages.forEach((msg) => {
      if (msg.role === "assistant" && msg.metadata) {
        const meta = msg.metadata as ResponseMetadata;
        currentActualCost += meta.cost;

        if (typeof gpt4CostPerToken === "number") {
          currentHypotheticalCost +=
            (meta.promptTokens + meta.completionTokens) * gpt4CostPerToken;
        } else {
        }
      }
    });

    setActualCost(currentActualCost);
    setHypotheticalCost(currentHypotheticalCost);
    if (typeof gpt4CostPerToken === "number") {
      const calculatedSavings = currentHypotheticalCost - currentActualCost;
      setTotalSavings(calculatedSavings);
      if (currentHypotheticalCost > 0) {
        setSavingsPercentage(
          (calculatedSavings / currentHypotheticalCost) * 100
        );
      } else {
        setSavingsPercentage(0);
      }
    } else {
      setTotalSavings(0);
      setSavingsPercentage(0);
    }
  }, [messages, modelCosts]);

  return (
    <aside className="w-full md:w-72 h-full border-r bg-background p-4 space-y-6 flex flex-col">
      <div>
        <div className="text-lg font-semibold mb-4">Router Settings</div>
        <div className="space-y-4">
          <StrategySelector value={strategy} onChange={onStrategyChange} />
          <Separator />
          <DeveloperModeToggle
            value={developerMode}
            onChange={onDeveloperModeChange}
          />
        </div>
      </div>
      <Separator />
      <div className="flex-grow">
        <div className="text-lg font-semibold mb-4">Chat Summary</div>
        <ChatSummary messages={messages} />
        <div className="mt-4 p-3 bg-muted/50 rounded-lg">
          <h4 className="text-sm font-medium mb-2">Cost Savings Tracker</h4>
          <div className="text-xs space-y-1">
            <div className="flex justify-between">
              <span>Actual Cost:</span>
              <span>${actualCost.toFixed(5)}</span>
            </div>
            <div className="flex justify-between">
              <span>Est. GPT-4 Cost:</span>
              <span>${hypotheticalCost.toFixed(5)}</span>
            </div>
            <div className="flex justify-between font-semibold">
              <span
                className={
                  totalSavings >= 0 ? "text-green-600" : "text-red-600"
                }
              >
                Savings:
              </span>
              <span
                className={
                  totalSavings >= 0 ? "text-green-600" : "text-red-600"
                }
              >
                ${totalSavings.toFixed(5)}
                {hypotheticalCost > 0 &&
                  savingsPercentage !== 0 &&
                  ` (${savingsPercentage.toFixed(1)}%)`}
              </span>
            </div>
            {modelCosts[GPT4_MODEL_ID]?.costPerToken === undefined && (
              <p className="text-red-500 text-xs mt-1">
                GPT-4 cost data missing, savings may be inaccurate.
              </p>
            )}
          </div>
        </div>
      </div>
    </aside>
  );
}
