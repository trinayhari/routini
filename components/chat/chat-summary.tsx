import { ChatMessage } from "@/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { DollarSign, Hash, MessageSquare } from "lucide-react";

interface ChatSummaryProps {
  messages: ChatMessage[];
}

export function ChatSummary({ messages }: ChatSummaryProps) {
  // Calculate total cost
  const totalCost = messages.reduce((sum, msg) => {
    if (msg.metadata?.cost) {
      return sum + msg.metadata.cost;
    }
    return sum;
  }, 0);

  // Calculate total tokens
  const totalTokens = messages.reduce((sum, msg) => {
    if (msg.metadata?.totalTokens) {
      return sum + msg.metadata.totalTokens;
    }
    return sum;
  }, 0);

  // Calculate total input tokens
  const totalPromptTokens = messages.reduce((sum, msg) => {
    if (msg.metadata?.promptTokens) {
      return sum + msg.metadata.promptTokens;
    }
    return sum;
  }, 0);

  // Calculate total output tokens
  const totalCompletionTokens = messages.reduce((sum, msg) => {
    if (msg.metadata?.completionTokens) {
      return sum + msg.metadata.completionTokens;
    }
    return sum;
  }, 0);

  // Get unique prompts (user messages)
  const uniquePrompts = messages
    .filter((msg) => msg.role === "user")
    .map((msg) => msg.content);

  return (
    <div className="space-y-2">
      <div className="text-sm font-medium">Chat Summary</div>
      <div className="space-y-3">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-2">
            <DollarSign className="h-4 w-4 text-muted-foreground" />
            <span>Total Cost</span>
          </div>
          <span className="font-medium">${totalCost.toFixed(6)}</span>
        </div>
        <div className="flex flex-col space-y-1">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <Hash className="h-4 w-4 text-muted-foreground" />
              <span>Total Tokens</span>
            </div>
            <span className="font-medium">{totalTokens.toLocaleString()}</span>
          </div>
          <div className="text-xs text-muted-foreground pl-6">
            <div className="flex justify-between">
              <span>Input:</span>
              <span>{totalPromptTokens.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span>Output:</span>
              <span>{totalCompletionTokens.toLocaleString()}</span>
            </div>
          </div>
        </div>
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-2">
            <MessageSquare className="h-4 w-4 text-muted-foreground" />
            <span>Unique Prompts</span>
          </div>
          <span className="font-medium">{uniquePrompts.length}</span>
        </div>

        {uniquePrompts.length > 0 && (
          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="prompts" className="border-none">
              <AccordionTrigger className="text-sm py-2 hover:no-underline">
                View Prompts
              </AccordionTrigger>
              <AccordionContent>
                <div className="space-y-2 text-sm">
                  {uniquePrompts.map((prompt, index) => (
                    <div
                      key={index}
                      className="p-2 bg-muted rounded-md text-muted-foreground"
                    >
                      {prompt}
                    </div>
                  ))}
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        )}
      </div>
    </div>
  );
}
