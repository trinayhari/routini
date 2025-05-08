// Force dynamic rendering for this API route
export const dynamic = 'force-dynamic';

import { NextResponse } from 'next/server';
import { GenerateRequest, GenerateResponse } from '@/types';

export async function POST(request: Request) {
  const start = Date.now();
  const requestId = `req_${start}`;
  
  try {
    const body: GenerateRequest = await request.json();
    
    console.log(`[${requestId}] Generate API request received:`, 
      { prompt: body.prompt?.substring(0, 50) + '...', strategy: body.strategy });
    
    try {
      // Call the FastAPI backend directly
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
      console.log(`[${requestId}] Connecting to backend at: ${backendUrl}`);
      
      // Start timer for backend request
      const backendStart = Date.now();
      
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
          messages: body.previousMessages ? body.previousMessages.map(msg => ({
            ...msg,
            metadata: msg.metadata ? JSON.stringify(msg.metadata) : undefined
          })) : undefined
        }),
      });
      
      const backendDuration = Date.now() - backendStart;
      console.log(`[${requestId}] Backend request completed in ${backendDuration}ms`);
      
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
        
        console.error(`[${requestId}] Backend error: ${response.status}`, errorMessage);
        throw new Error(`Backend returned ${response.status}: ${errorMessage}`);
      }
      
      // Parse and handle response
      const parseStart = Date.now();
      let backendResponse;
      try {
        backendResponse = await response.json();
        const parseDuration = Date.now() - parseStart;
        console.log(`[${requestId}] Response parsed in ${parseDuration}ms`);
      } catch (parseError) {
        console.error(`[${requestId}] Error parsing response JSON:`, parseError);
        throw new Error('Invalid JSON response from backend');
      }
      
      if (!backendResponse) {
        console.error(`[${requestId}] Invalid response format:`, backendResponse);
        throw new Error('Invalid response format from backend');
      }
      
      // Convert the backend response to frontend format
      const response_data: GenerateResponse = {
        message: {
          id: Date.now().toString(),
          role: 'assistant',
          content: backendResponse.text,
          timestamp: new Date(),
          metadata: {
            model: backendResponse.model,
            latencyMs: backendResponse.latency * 1000,
            promptTokens: backendResponse.prompt_tokens,
            completionTokens: backendResponse.completion_tokens,
            totalTokens: backendResponse.tokens,
            cost: backendResponse.cost,
            routingRationale: backendResponse.classification?.selected_reason
          }
        }
      };
      
      const totalDuration = Date.now() - start;
      console.log(`[${requestId}] Total request completed in ${totalDuration}ms`);
      
      return NextResponse.json(response_data);
    } catch (backendError) {
      console.error(`[${requestId}] Backend API error:`, backendError);
      
      // Return a more detailed error for debugging
      return NextResponse.json(
        { 
          error: 'FastAPI backend error', 
          message: backendError instanceof Error ? backendError.message : String(backendError),
          prompt: body.prompt,
          requestId
        },
        { status: 500 }
      );
    }
  } catch (error) {
    console.error(`[${requestId}] Error in generate API:`, error);
    return NextResponse.json(
      { error: 'Failed to generate response', details: String(error), requestId },
      { status: 500 }
    );
  }
}