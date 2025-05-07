// Routing strategy types
export type RoutingStrategy = 'cost' | 'quality' | 'balanced';

// Message types
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

// Model response metadata
export interface ModelMetadata {
  latencyMs: number;
  totalTokens: number;
  promptTokens: number;
  completionTokens: number;
  cost: number;
  model: string;
  routingRationale?: string;
}

// Chat message with optional metadata
export interface ChatMessage extends Message {
  metadata?: ModelMetadata;
}

// For developer mode trace information
export interface RoutingTrace {
  timestamp: string;
  inputTokenEstimate: number;
  strategy: RoutingStrategy;
  candidateModels: Array<{
    name: string;
    score: number;
    costPerToken: number;
    qualityScore: number;
    latencyScore: number;
  }>;
  selectedModel: string;
  reason: string;
}

// API request/response types
export interface GenerateRequest {
  prompt: string;
  model?: string;
  messages?: Array<{ role: string; content: string }>;
  max_tokens?: number;
  temperature?: number;
}

export interface GenerateResponse {
  text: string;
  metadata: ModelMetadata;
}

// Compare models
export interface CompareRequest {
  prompt: string;
  models: string[];
}

export interface ModelComparison {
  model: string;
  response: string;
  metadata: ModelMetadata;
}

export interface CompareResponse {
  results: ModelComparison[];
}