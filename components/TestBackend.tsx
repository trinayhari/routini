"use client";

import { useState } from "react";
import { generateResponse } from "@/lib/api";
import { RoutingStrategy } from "@/types";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
  CardDescription,
} from "./ui/card";
import { Loader2 } from "lucide-react";

export default function TestBackend() {
  const [prompt, setPrompt] = useState("");
  const [strategy, setStrategy] = useState<RoutingStrategy>("balanced");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    if (!prompt.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await generateResponse({
        prompt,
        strategy,
      });

      setResult(response);
    } catch (err) {
      console.error("Error sending request:", err);
      setError(
        err instanceof Error ? err.message : "An unknown error occurred"
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container mx-auto py-8 max-w-4xl">
      <h1 className="text-2xl font-bold mb-6">Test Backend Integration</h1>

      <div className="grid gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Send Prompt to Backend</CardTitle>
            <CardDescription>
              This will route your prompt to the best model based on the
              selected strategy
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="block mb-2 text-sm font-medium">Prompt</label>
              <Textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter your prompt here..."
                className="min-h-[120px]"
              />
            </div>

            <div>
              <label className="block mb-2 text-sm font-medium">
                Routing Strategy
              </label>
              <Select
                value={strategy}
                onValueChange={(value) => setStrategy(value as RoutingStrategy)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select a strategy" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="balanced">Balanced</SelectItem>
                  <SelectItem value="cost">Cost-Optimized</SelectItem>
                  <SelectItem value="quality">Quality-Optimized</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
          <CardFooter>
            <Button
              onClick={handleSubmit}
              disabled={loading || !prompt.trim()}
              className="w-full"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                "Generate Response"
              )}
            </Button>
          </CardFooter>
        </Card>

        {error && (
          <Card className="border-red-300 bg-red-50">
            <CardContent className="pt-6">
              <p className="text-red-600">{error}</p>
            </CardContent>
          </Card>
        )}

        {result && (
          <Card>
            <CardHeader>
              <CardTitle>Response</CardTitle>
              <CardDescription>
                Routed to: {result.message.metadata?.model || "Unknown model"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="whitespace-pre-wrap bg-gray-50 p-4 rounded-md">
                {result.message.content}
              </div>

              {result.message.metadata && (
                <div className="mt-6 grid gap-2 text-sm">
                  <div className="grid grid-cols-2 gap-2 border-b pb-2">
                    <span className="font-medium">Latency:</span>
                    <span>
                      {(result.message.metadata.latencyMs / 1000).toFixed(2)}{" "}
                      seconds
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 border-b pb-2">
                    <span className="font-medium">Cost:</span>
                    <span>${result.message.metadata.cost.toFixed(6)}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 border-b pb-2">
                    <span className="font-medium">Tokens:</span>
                    <span>{result.message.metadata.totalTokens}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 border-b pb-2 text-sm pl-6">
                    <span className="text-muted-foreground">Input Tokens:</span>
                    <span>{result.message.metadata.promptTokens}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 border-b pb-2 text-sm pl-6">
                    <span className="text-muted-foreground">
                      Output Tokens:
                    </span>
                    <span>{result.message.metadata.completionTokens}</span>
                  </div>
                  <div className="mt-2">
                    <span className="font-medium block mb-1">
                      Routing Explanation:
                    </span>
                    <p className="text-gray-700">
                      {result.message.metadata.routingRationale ||
                        "No explanation provided"}
                    </p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
