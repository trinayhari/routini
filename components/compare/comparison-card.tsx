import { ModelComparison } from "@/types";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { MessageContent } from "@/components/chat/message-content";
import { cn } from "@/lib/utils";
import { Clock, DollarSign, Bot, Hash } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface ComparisonCardProps {
  result: ModelComparison;
  fullWidth?: boolean;
}

export function ComparisonCard({
  result,
  fullWidth = false,
}: ComparisonCardProps) {
  const { model, response, metadata } = result;

  const formattedCost = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 5,
    maximumFractionDigits: 5,
  }).format(metadata.cost);

  return (
    <Card className={cn("h-full flex flex-col", fullWidth ? "w-full" : "")}>
      <CardHeader className="pb-2">
        <CardTitle className="text-md flex items-center gap-2">
          <Bot className="h-4 w-4" />
          {model}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-y-auto">
        <MessageContent content={response} />
      </CardContent>
      <CardFooter className="border-t pt-4 text-sm text-muted-foreground flex justify-between">
        <div className="flex items-center gap-1">
          <Clock className="h-4 w-4" />
          <span>{(metadata.latencyMs / 1000).toFixed(1)}s</span>
        </div>
        <div className="flex items-center gap-1">
          <DollarSign className="h-4 w-4" />
          <span>{formattedCost}</span>
        </div>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger className="flex items-center gap-1">
              <Hash className="h-4 w-4" />
              <span>{metadata.totalTokens} tokens</span>
            </TooltipTrigger>
            <TooltipContent>
              <div className="text-xs space-y-1">
                <div className="flex justify-between gap-4">
                  <span>Input:</span>
                  <span>{metadata.promptTokens} tokens</span>
                </div>
                <div className="flex justify-between gap-4">
                  <span>Output:</span>
                  <span>{metadata.completionTokens} tokens</span>
                </div>
              </div>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </CardFooter>
    </Card>
  );
}
