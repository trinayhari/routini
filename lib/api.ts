import {
  GenerateRequest,
  GenerateResponse,
  CompareRequest,
  CompareResponse,
} from '@/types';

/**
 * Send a prompt to the LLM router backend (Next.js API route)
 */
export async function generateResponse(
  request: GenerateRequest
): Promise<GenerateResponse> {
  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return response.json();
  } catch (error) {
    console.error('Error generating response:', error);
    throw error;
  }
}

/**
 * Compare the same prompt across multiple models
 */
export async function compareModels(
  request: CompareRequest
): Promise<CompareResponse> {
  try {
    const response = await fetch('/api/compare', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return response.json();
  } catch (error) {
    console.error('Error comparing models:', error);
    throw error;
  }
}

/**
 * Interface for requests to the FastAPI backend
 */
export interface FastAPIRequest {
  prompt: string;
  routing_strategy: 'fastest' | 'cheapest' | 'most_capable' | 'balanced';
  max_tokens?: number;
  temperature?: number;
}

/**
 * Interface for responses from the FastAPI backend
 */
export interface FastAPIResponse {
  model_used: string;
  response: string;
  latency_seconds: number;
  estimated_cost: number;
  routing_explanation: string;
}

/**
 * Send a prompt directly to the FastAPI backend
 */
export async function sendPrompt(
  prompt: string,
  strategy: 'fastest' | 'cheapest' | 'most_capable' | 'balanced' = 'balanced',
  maxTokens: number = 1024,
  temperature: number = 0.7
): Promise<FastAPIResponse> {
  try {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
    
    const response = await fetch(`${backendUrl}/generate/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt,
        routing_strategy: strategy,
        max_tokens: maxTokens,
        temperature
      } as FastAPIRequest),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`FastAPI backend error: ${response.status} - ${errorText}`);
    }

    return await response.json() as FastAPIResponse;
  } catch (error) {
    console.error('Error sending prompt to FastAPI backend:', error);
    throw error;
  }
}

/**
 * Interface for FastAPI compare request
 */
export interface FastAPICompareRequest {
  prompt: string;
  models?: string[];
  max_tokens?: number;
  temperature?: number;
}

/**
 * Interface for model response in FastAPI compare response
 */
export interface FastAPIModelResponse {
  model: string;
  response: string;
  metadata: {
    model: string;
    latencyMs: number;
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    cost: number;
  };
}

/**
 * Interface for FastAPI compare response
 */
export interface FastAPICompareResponse {
  results: FastAPIModelResponse[];
  prompt: string;
}

/**
 * Compare a prompt across multiple models using the FastAPI backend
 */
export async function compareWithBackend(
  prompt: string,
  models?: string[],
  maxTokens: number = 1024,
  temperature: number = 0.7
): Promise<FastAPICompareResponse> {
  try {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
    console.log(`Connecting to backend at: ${backendUrl}`);
    
    // Create the request body
    const requestBody = {
      prompt,
      models,
      max_tokens: maxTokens,
      temperature
    };
    
    console.log('Request payload:', JSON.stringify(requestBody));
    
    // Make the API call
    const response = await fetch(`${backendUrl}/compare/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    // Handle non-200 responses
    if (!response.ok) {
      let errorText;
      try {
        // Try to parse as JSON
        const errorJson = await response.json();
        errorText = JSON.stringify(errorJson);
      } catch (e) {
        // Fall back to text if not JSON
        errorText = await response.text();
      }
      
      throw new Error(`FastAPI backend error: ${response.status} - ${errorText}`);
    }

    // Parse the response
    const data = await response.json();
    console.log('Response data:', data);
    return data as FastAPICompareResponse;
  } catch (error) {
    console.error('Error comparing with FastAPI backend:', error);
    // Rethrow to allow the calling code to handle it
    throw error;
  }
}