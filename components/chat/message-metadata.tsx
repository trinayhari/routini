import { FC, useEffect, useState } from "react";
import { ResponseMetadata, RoutingTrace } from "@/types";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Clock, Sparkles, Zap, Scale, Router, DollarSign } from "lucide-react";
import { formatDistanceToNow } from "date-fns";

interface MessageMetadataProps {
  metadata: ResponseMetadata;
  trace?: RoutingTrace;
  showTrace?: boolean;
}

export const MessageMetadata: FC<MessageMetadataProps> = ({
  metadata,
  trace,
  showTrace = false,
}) => {
  const formattedCost = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 5,
    maximumFractionDigits: 5,
  }).format(metadata.cost);

  const [modelChanged, setModelChanged] = useState(false);

  useEffect(() => {
    if (metadata.model) {
      setModelChanged(true);
      const timer = setTimeout(() => setModelChanged(false), 500); // Duration of the animation
      return () => clearTimeout(timer);
    }
  }, [metadata.model]);

  return (
    <div className="inline-flex items-center gap-1.5 rounded-full px-2 py-1 text-xs bg-muted/30">
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger>
            <span
              className={`font-semibold ${
                modelChanged ? "animate-model-change" : ""
              }`}
            >
              {metadata.model}
            </span>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p>Model: {metadata.model}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      <div className="h-3 w-[1px] bg-border"></div>

      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger className="inline-flex items-center">
            <Clock className="h-3 w-3 mr-1" />
            <span>{(metadata.latencyMs / 1000).toFixed(1)}s</span>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p>Response time: {metadata.latencyMs}ms</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      <div className="h-3 w-[1px] bg-border"></div>

      <HoverCard>
        <HoverCardTrigger className="inline-flex items-center">
          <DollarSign className="h-3 w-3 mr-1" />
          <span>{formattedCost}</span>
        </HoverCardTrigger>
        <HoverCardContent className="w-80" side="bottom">
          <div className="flex flex-col space-y-1">
            <div className="text-sm font-semibold">Cost breakdown</div>
            <div className="text-xs grid grid-cols-2 gap-2">
              <span className="text-muted-foreground">Prompt tokens:</span>
              <span>{metadata.promptTokens}</span>
              <span className="text-muted-foreground">Completion tokens:</span>
              <span>{metadata.completionTokens}</span>
              <span className="text-muted-foreground">Total tokens:</span>
              <span>{metadata.totalTokens}</span>
              <span className="text-muted-foreground">Total cost:</span>
              <span>{formattedCost}</span>
            </div>
          </div>
        </HoverCardContent>
      </HoverCard>

      {metadata.routingRationale && (
        <>
          <div className="h-3 w-[1px] bg-border"></div>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger className="inline-flex items-center">
                <Router className="h-3 w-3" />
              </TooltipTrigger>
              <TooltipContent side="bottom" className="max-w-xs">
                <p className="text-xs">{metadata.routingRationale}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </>
      )}

      {showTrace && trace && (
        <div className="mt-2 w-full">
          <div className="bg-muted/30 rounded-md p-3 text-xs font-mono overflow-x-auto">
            <div className="font-semibold mb-1">Routing Trace</div>
            <pre className="whitespace-pre-wrap break-all">
              {JSON.stringify(trace, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};
