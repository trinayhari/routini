// Force dynamic rendering for this API route
export const dynamic = 'force-dynamic';

import { NextResponse } from 'next/server';
import { GenerateRequest, GenerateResponse } from '@/types';

export async function POST(request: Request) {
  try {
    const body: GenerateRequest = await request.json();
    
    console.log('Generate API request received:', body);
    
    try {
      // Call the FastAPI backend directly
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
      console.log(`Connecting directly to backend at: ${backendUrl}`);
      
      const response = await fetch(`${backendUrl}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: body.prompt,
          routing_strategy: body.strategy,
          max_tokens: body.max_tokens || 1024,
          temperature: body.temperature || 0.7,
          messages: body.previousMessages
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
      
      if (!backendResponse) {
        console.error('Invalid response format:', backendResponse);
        throw new Error('Invalid response format from backend');
      }
      
      // Convert the backend response to frontend format
      const response_data: GenerateResponse = {
        message: {
          id: Date.now().toString(),
          role: 'assistant',
          content: backendResponse.response,
          timestamp: new Date(),
          metadata: {
            model: backendResponse.model_used,
            latencyMs: backendResponse.latency_seconds * 1000,
            promptTokens: backendResponse.token_metrics.prompt,
            completionTokens: backendResponse.token_metrics.completion,
            totalTokens: backendResponse.token_metrics.total,
            cost: backendResponse.estimated_cost,
            routingRationale: backendResponse.routing_explanation
          }
        }
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
    console.error('Error in generate API:', error);
    return NextResponse.json(
      { error: 'Failed to generate response', details: String(error) },
      { status: 500 }
    );
  }
}