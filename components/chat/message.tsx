import { ChatMessage } from "@/types";
import { cn } from "@/lib/utils";
import { format } from "date-fns";
import { MessageMetadata } from "@/components/chat/message-metadata";
import { MessageContent } from "@/components/chat/message-content";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { ChevronDown } from "lucide-react";

interface MessageProps {
  message: ChatMessage;
  showTrace?: boolean;
}

export function Message({ message, showTrace = false }: MessageProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex w-full mb-4 animate-in fade-in-0 slide-in-from-bottom-3 duration-300",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div
        className={cn(
          "flex flex-col space-y-2 max-w-[80%] md:max-w-[70%]",
          isUser ? "items-end" : "items-start"
        )}
      >
        <div
          className={cn(
            "rounded-lg px-4 py-3 shadow-sm",
            isUser ? "bg-primary text-primary-foreground" : "bg-muted"
          )}
        >
          <MessageContent content={message.content} />
        </div>

        {!isUser && message.metadata && (
          <div className="space-y-2 w-full">
            <MessageMetadata
              metadata={message.metadata}
              showTrace={showTrace}
            />

            {message.metadata.routingRationale && (
              <Accordion type="single" collapsible className="w-full">
                <AccordionItem value="explanation" className="border-none">
                  <AccordionTrigger className="text-xs py-1 hover:no-underline">
                    <div className="flex items-center gap-1 text-muted-foreground">
                      <span>Why this model?</span>
                      <ChevronDown className="h-3 w-3" />
                    </div>
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="text-xs text-muted-foreground p-2 bg-muted rounded-md">
                      {message.metadata.routingRationale}
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            )}
          </div>
        )}

        <div className="flex items-center gap-2 text-xs text-muted-foreground px-1">
          {message.timestamp && (
            <time dateTime={message.timestamp.toString()}>
              {format(new Date(message.timestamp), "h:mm a")}
            </time>
          )}
        </div>
      </div>
    </div>
  );
}
