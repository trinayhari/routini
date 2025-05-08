"use client";

import { useState } from "react";
import { CompareResponse, ModelComparison, RoutingStrategy } from "@/types";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Loader2, RefreshCw } from "lucide-react";
import { MainLayout } from "@/components/layout/main-layout";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// List of available models
const AVAILABLE_MODELS = [
  {
    id: "anthropic/claude-3-haiku",
    name: "Claude 3 Haiku",
    description: "Fast and efficient model for routine tasks",
  },
  {
    id: "anthropic/claude-3-sonnet",
    name: "Claude 3 Sonnet",
    description: "Balanced model with strong reasoning and creativity",
  },
  {
    id: "anthropic/claude-3-opus",
    name: "Claude 3 Opus",
    description: "Most powerful model for complex tasks",
  },
  {
    id: "openai/gpt-4",
    name: "GPT-4",
    description: "Latest OpenAI model with balanced capabilities",
  },
  {
    id: "openai/gpt-3.5-turbo",
    name: "GPT-3.5 Turbo",
    description: "Fast and cost-effective model",
  },
  {
    id: "mistralai/mixtral-8x7b-instruct",
    name: "Mixtral 8x7B",
    description: "Powerful open-source mixture-of-experts model",
  },
  {
    id: "mistralai/mistral-7b-instruct",
    name: "Mistral 7B",
    description: "Compact and efficient open-source model",
  },
  {
    id: "meta-llama/llama-2-70b-chat",
    name: "Llama 2 70B",
    description: "Meta's latest large language model",
  },
];

// Get the default models to compare
const DEFAULT_MODELS = [
  "openai/gpt-4",
  "anthropic/claude-3-opus",
  "mistralai/mixtral-8x7b-instruct",
];

// Add interface for component props
interface ComparePageProps {
  strategy?: RoutingStrategy;
  developerMode?: boolean;
  onMessagesUpdate?: (messages: any[]) => void;
}

export default function ComparePage({
  strategy,
  developerMode,
  onMessagesUpdate,
}: ComparePageProps) {
  const [prompt, setPrompt] = useState("");
  const [results, setResults] = useState<ModelComparison[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Track which models are being compared
  const [selectedModels, setSelectedModels] =
    useState<string[]>(DEFAULT_MODELS);

  // State for managing alternative model selection
  const [alternativeModel, setAlternativeModel] = useState<string>("");
  const [replacingModel, setReplacingModel] = useState<string | null>(null);
  const [loadingAlternative, setLoadingAlternative] = useState(false);

  async function handleCompare() {
    if (!prompt.trim()) return;

    setLoading(true);
    setError(null);

    // Maximum number of retries
    const maxRetries = 2;
    let retries = 0;
    let lastError = null;

    const attemptCompare = async (): Promise<boolean> => {
      try {
        console.log(
          `Attempt ${
            retries + 1
          }: Sending request to compare models with prompt: "${prompt}"`
        );

        // Make request directly to the Next.js API route
        const response = await fetch("/api/compare", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            prompt,
            models: selectedModels,
          }),
        });

        if (!response.ok) {
          // Clone the response before reading
          const clonedResponse = response.clone();

          let errorMessage = `API error: ${response.status}`;

          try {
            // Try to parse as JSON
            const errorData = await response.json();
            errorMessage += ` - ${JSON.stringify(errorData)}`;
          } catch (jsonError) {
            try {
              // If not JSON, get as text from the clone
              const errorText = await clonedResponse.text();
              errorMessage += ` - ${errorText}`;
            } catch (textError) {
              // If both fail, use a generic message
              errorMessage += " - Could not read error details";
            }
          }

          throw new Error(errorMessage);
        }

        const data = await response.json();
        console.log("Response received:", data);

        if (!data || !data.results) {
          throw new Error("Invalid response format: missing results");
        }

        setResults(data.results);
        return true;
      } catch (err) {
        console.error(`Attempt ${retries + 1} error:`, err);
        lastError = err;

        if (retries < maxRetries) {
          retries++;
          console.log(`Retrying... (${retries}/${maxRetries})`);
          // Wait a second before retrying
          await new Promise((resolve) => setTimeout(resolve, 1000));
          return await attemptCompare();
        }

        // If we've exhausted retries, set the error
        const errorMessage =
          err instanceof Error ? err.message : "An unknown error occurred";
        setError(errorMessage);
        return false;
      }
    };

    try {
      await attemptCompare();
    } finally {
      setLoading(false);
    }
  }

  // Function to retry with an alternative model
  async function handleRetryWithAlternative(modelToReplace: string) {
    if (
      !prompt.trim() ||
      !alternativeModel ||
      alternativeModel === modelToReplace
    )
      return;

    setLoadingAlternative(true);
    setReplacingModel(modelToReplace);

    try {
      // Create new model list by replacing the selected model
      const newModels = selectedModels.map((model) =>
        model === modelToReplace ? alternativeModel : model
      );

      // Update the selected models
      setSelectedModels(newModels);

      // Make request to the Next.js API route
      const response = await fetch("/api/compare", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt,
          models: [alternativeModel],
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();

      if (!data || !data.results || data.results.length === 0) {
        throw new Error("Invalid response format or empty results");
      }

      // Find and replace the model in results
      if (results) {
        const newResults = results.map((result) => {
          if (result.model === modelToReplace) {
            return data.results[0];
          }
          return result;
        });

        setResults(newResults);
      }
    } catch (err) {
      console.error("Error retrying with alternative model:", err);
      setError(
        `Failed to get response from ${alternativeModel}: ${
          err instanceof Error ? err.message : "Unknown error"
        }`
      );
    } finally {
      setLoadingAlternative(false);
      setReplacingModel(null);
    }
  }

  return (
    <MainLayout>
      <div className="container mx-auto py-8 max-w-7xl">
        <h1 className="text-3xl font-bold mb-6">Model Comparison</h1>

        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Compare LLM Responses</CardTitle>
            <CardDescription>
              Enter a prompt to see how different models respond
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter your prompt here..."
              className="min-h-[120px]"
            />

            <div className="mt-4 flex flex-col space-y-2">
              <h3 className="text-sm font-medium">Models to compare:</h3>
              <div className="flex flex-wrap gap-2">
                {selectedModels.map((modelId) => {
                  const model = AVAILABLE_MODELS.find(
                    (m) => m.id === modelId
                  ) || { id: modelId, name: modelId, description: "Unknown" };
                  return (
                    <div
                      key={modelId}
                      className="inline-flex items-center rounded-full bg-secondary px-2.5 py-0.5 text-xs font-medium"
                    >
                      {model.name}
                    </div>
                  );
                })}
              </div>
            </div>
          </CardContent>
          <CardFooter>
            <Button
              onClick={handleCompare}
              disabled={loading || !prompt.trim()}
              className="w-full"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Comparing Models...
                </>
              ) : (
                "Compare Models"
              )}
            </Button>
          </CardFooter>
        </Card>

        {error && (
          <Card className="border-red-300 bg-red-50 mb-8">
            <CardContent className="pt-6">
              <p className="text-red-600 font-medium">Error: {error}</p>
              <div className="mt-4">
                <p className="text-sm text-red-600">
                  Please make sure the FastAPI backend is running at
                  http://localhost:8000
                </p>
                <p className="text-sm text-red-600 mt-1">
                  Try running: <code>cd backend && python run.py</code>
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {results && results.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {results.map((result, index) => (
              <ModelResponseCard
                key={index}
                result={result}
                onRetryWithAlternative={(modelId) => {
                  setAlternativeModel("");
                  setReplacingModel(modelId);
                }}
                isReplacing={replacingModel === result.model}
                loadingAlternative={
                  loadingAlternative && replacingModel === result.model
                }
                alternativeModel={alternativeModel}
                onAlternativeModelChange={setAlternativeModel}
                onConfirmRetry={() => handleRetryWithAlternative(result.model)}
                availableModels={AVAILABLE_MODELS.filter(
                  (model) =>
                    !selectedModels.includes(model.id) ||
                    model.id === alternativeModel
                )}
              />
            ))}
          </div>
        )}
      </div>
    </MainLayout>
  );
}

interface ModelResponseCardProps {
  result: ModelComparison;
  onRetryWithAlternative: (modelId: string) => void;
  isReplacing: boolean;
  loadingAlternative: boolean;
  alternativeModel: string;
  onAlternativeModelChange: (modelId: string) => void;
  onConfirmRetry: () => void;
  availableModels: Array<{ id: string; name: string; description: string }>;
}

function ModelResponseCard({
  result,
  onRetryWithAlternative,
  isReplacing,
  loadingAlternative,
  alternativeModel,
  onAlternativeModelChange,
  onConfirmRetry,
  availableModels,
}: ModelResponseCardProps) {
  // Function to get a friendly model name
  const getModelName = (modelId: string) => {
    if (modelId.includes("gpt-4o")) return "GPT-4o";
    if (modelId.includes("claude-3-opus")) return "Claude 3 Opus";
    if (modelId.includes("mixtral")) return "Mixtral 8x7B";
    return modelId.split("/").pop() || modelId;
  };

  return (
    <Card className="flex flex-col h-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex items-center justify-between">
          <span>{getModelName(result.model)}</span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onRetryWithAlternative(result.model)}
          >
            <RefreshCw className="h-4 w-4 mr-1" />
            <span className="sr-only md:not-sr-only md:text-xs">
              Retry with another
            </span>
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-grow relative">
        {isReplacing && (
          <div className="absolute inset-0 bg-background/80 backdrop-blur-sm z-10 flex flex-col items-center justify-center p-4">
            <div className="mb-4 text-center">
              <h3 className="font-medium mb-2">Replace with another model</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Select a different model to compare using the same prompt
              </p>

              <Select
                value={alternativeModel}
                onValueChange={onAlternativeModelChange}
              >
                <SelectTrigger className="w-full mb-4">
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {availableModels.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <div className="flex flex-col space-y-2">
                <Button
                  onClick={onConfirmRetry}
                  disabled={!alternativeModel || loadingAlternative}
                >
                  {loadingAlternative ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Loading...
                    </>
                  ) : (
                    "Retry with selected model"
                  )}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => onRetryWithAlternative("")}
                  disabled={loadingAlternative}
                >
                  Cancel
                </Button>
              </div>
            </div>
          </div>
        )}
        <div className="whitespace-pre-wrap bg-gray-50 p-4 rounded-md max-h-[500px] overflow-y-auto">
          {result.response}
        </div>
      </CardContent>
      <CardFooter className="flex flex-col items-start border-t pt-4">
        <div className="w-full space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="font-medium">Latency:</span>
            <span>{(result.metadata.latencyMs / 1000).toFixed(2)} seconds</span>
          </div>
          <div className="flex flex-col space-y-1">
            <div className="flex justify-between">
              <span className="font-medium">Cost:</span>
              <span>${result.metadata.cost.toFixed(6)}</span>
            </div>
            <div className="text-xs text-muted-foreground pl-2">
              <div className="flex justify-between">
                <span>Input:</span>
                <span>
                  $
                  {(
                    result.metadata.promptTokens *
                    (result.metadata.cost / result.metadata.totalTokens)
                  ).toFixed(6)}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Output:</span>
                <span>
                  $
                  {(
                    result.metadata.completionTokens *
                    (result.metadata.cost / result.metadata.totalTokens)
                  ).toFixed(6)}
                </span>
              </div>
            </div>
          </div>
          <div className="flex flex-col space-y-1">
            <div className="flex justify-between">
              <span className="font-medium">Tokens:</span>
              <span>{result.metadata.totalTokens}</span>
            </div>
            <div className="text-xs text-muted-foreground pl-2">
              <div className="flex justify-between">
                <span>Input:</span>
                <span>{result.metadata.promptTokens}</span>
              </div>
              <div className="flex justify-between">
                <span>Output:</span>
                <span>{result.metadata.completionTokens}</span>
              </div>
            </div>
          </div>
        </div>
      </CardFooter>
    </Card>
  );
}
