// Force dynamic rendering for this API route
export const dynamic = 'force-dynamic';

import { NextResponse } from 'next/server';
import { CompareRequest, CompareResponse, ModelComparison } from '@/types';
import { FastAPIModelResponse } from '@/lib/api';

// Default models to compare
const DEFAULT_MODELS = [
  "openai/gpt-4",
  "anthropic/claude-3-opus",
  "mistralai/mixtral-8x7b-instruct"
];

export async function POST(request: Request) {
  try {
    const body: CompareRequest = await request.json();
    
    console.log('Compare API request received:', body);
    
    try {
      // Call the FastAPI backend directly
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
      console.log(`Connecting directly to backend at: ${backendUrl}`);
      
      const response = await fetch(`${backendUrl}/compare`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: body.prompt,
          models: body.models || DEFAULT_MODELS,
          max_tokens: body.max_tokens || 1024,
          temperature: body.temperature || 0.7
        }),
      });
      
      if (!response.ok) {
        let errorMessage;
        
        try {
          // Clone the response before reading
          const clonedResponse = response.clone();
          
          try {
            const errorJson = await response.json();
            errorMessage = JSON.stringify(errorJson);
          } catch (jsonError) {
            // If JSON parsing fails, try text
            errorMessage = await clonedResponse.text();
          }
        } catch (readError) {
          // If all reads fail, use a generic message
          errorMessage = 'Unable to read error details';
        }
        
        throw new Error(`Backend returned ${response.status}: ${errorMessage}`);
      }
      
      // Parse and handle response
      let backendResponse;
      try {
        backendResponse = await response.json();
        console.log('FastAPI backend response received:', backendResponse);
      } catch (parseError) {
        console.error('Error parsing response JSON:', parseError);
        throw new Error('Invalid JSON response from backend');
      }
      
      if (!backendResponse || !backendResponse.results) {
        console.error('Invalid response format:', backendResponse);
        throw new Error('Invalid response format from backend');
      }
      
      // Convert the backend response to frontend format
      const results: ModelComparison[] = backendResponse.results.map(
        (modelResponse: FastAPIModelResponse) => {
          // Ensure all metadata fields are present and have default values if missing
          const metadata = {
            model: modelResponse.model,
            latencyMs: modelResponse.metadata.latencyMs || 0,
            promptTokens: modelResponse.metadata.promptTokens || 0,
            completionTokens: modelResponse.metadata.completionTokens || 0,
            totalTokens: modelResponse.metadata.totalTokens || 0,
            cost: modelResponse.metadata.cost || 0
          };
          
          return {
            model: modelResponse.model,
            response: modelResponse.response,
            metadata
          };
        }
      );
      
      const response_data: CompareResponse = {
        results,
        originalPrompt: body.prompt
      };
      
      return NextResponse.json(response_data);
    } catch (backendError) {
      console.error('Backend API error details:', backendError);
      
      // Return a more detailed error for debugging
      return NextResponse.json(
        { 
          error: 'FastAPI backend error', 
          message: backendError instanceof Error ? backendError.message : String(backendError),
          prompt: body.prompt
        },
        { status: 500 }
      );
    }
  } catch (error) {
    console.error('Error in compare API:', error);
    return NextResponse.json(
      { error: 'Failed to compare models', details: String(error) },
      { status: 500 }
    );
  }
}