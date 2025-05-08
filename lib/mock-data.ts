import { 
  ChatMessage, 
  RoutingStrategy, 
  ModelComparison, 
  RoutingTrace,
  ModelMetadata
} from '@/types';

// Sample model configurations
export const MODELS = {
  'openai/gpt-4': { 
    name: 'GPT-4',
    costPerToken: 0.00006,
    qualityScore: 0.95,
  },
  'openai/gpt-3.5-turbo': { 
    name: 'GPT-3.5 Turbo',
    costPerToken: 0.000002,
    qualityScore: 0.82,
  },
  'anthropic/claude-2': { 
    name: 'Claude 2',
    costPerToken: 0.00008, 
    qualityScore: 0.93,
  },
  'meta-llama/llama-2-70b': { 
    name: 'Llama 2 (70B)',
    costPerToken: 0.000008,
    qualityScore: 0.85,
  },
  'mistralai/mistral-7b': { 
    name: 'Mistral 7B',
    costPerToken: 0.000005,
    qualityScore: 0.78,
  }
};

// Mock conversation history
export const MOCK_MESSAGES: ChatMessage[] = [
  {
    id: '1',
    role: 'user',
    content: 'Can you explain how LLM routing works?',
    timestamp: new Date(Date.now() - 1000 * 60 * 5),
  },
  {
    id: '2',
    role: 'assistant',
    content: 
      'LLM routing is the process of selecting the most appropriate language model for a given query. It works by analyzing the input prompt and making decisions based on factors like:' +
      '\n\n1. **Complexity** of the request' +
      '\n2. **Length** of expected response' + 
      '\n3. **Specialized knowledge** requirements' +
      '\n4. **Cost efficiency** considerations' +
      '\n\nThe router examines these factors and routes your query to the model that best balances performance and cost based on your selected strategy.',
    timestamp: new Date(Date.now() - 1000 * 60 * 4),
    metadata: {
      model: 'openai/gpt-3.5-turbo',
      latencyMs: 850,
      promptTokens: 12,
      completionTokens: 128,
      totalTokens: 140,
      cost: 0.00028,
      routingRationale: 'Selected for general knowledge query with moderate complexity'
    }
  },
  {
    id: '3',
    role: 'user',
    content: 'What are the advantages of using a router instead of directly calling a specific model?',
    timestamp: new Date(Date.now() - 1000 * 60 * 3),
  },
  {
    id: '4',
    role: 'assistant',
    content: 
      'Using an LLM router instead of directly calling a specific model offers several key advantages:' +
      '\n\n**Cost Optimization**' +
      '\n- Only use expensive models when necessary' +
      '\n- Automatically downgrade to cheaper models for simpler tasks' +
      '\n- Significant cost savings for high-volume applications' +
      '\n\n**Performance Improvement**' +
      '\n- Select models with specialized capabilities for specific tasks' +
      '\n- Maintain quality while minimizing latency' +
      '\n- Intelligently balance response time vs quality' +
      '\n\n**Developer Experience**' +
      '\n- Single API regardless of underlying model' +
      '\n- No need to manage multiple provider integrations' +
      '\n- Automatic fallbacks if a provider is unavailable' +
      '\n\n**Future-Proofing**' +
      '\n- Easy integration of new models as they become available' +
      '\n- Abstract provider-specific details away from application code' +
      '\n- Adaptable to evolving LLM ecosystem',
    timestamp: new Date(Date.now() - 1000 * 60 * 2),
    metadata: {
      model: 'openai/gpt-4',
      latencyMs: 2100,
      promptTokens: 18,
      completionTokens: 245,
      totalTokens: 263,
      cost: 0.01578,
      routingRationale: 'Selected for detailed explanation requiring comprehensive understanding'
    }
  }
];

// Create mock routing trace based on strategy
export function createMockTrace(
  prompt: string, 
  strategy: RoutingStrategy,
  selectedModel: string = 'openai/gpt-3.5-turbo'
): RoutingTrace {
  const tokenEstimate = Math.floor(prompt.length / 4) + 10;
  
  // Generate candidate model scores based on strategy
  const candidateModels = Object.entries(MODELS).map(([id, model]) => {
    let score = 0;
    
    if (strategy === 'cost') {
      score = 1 / (model.costPerToken * 100); // Invert so lower cost = higher score
    } else if (strategy === 'quality') {
      score = model.qualityScore * 10;
    } else {
      // Balanced approach
      score = (model.qualityScore * 5) + (1 / (model.costPerToken * 50));
    }
    
    // Add some randomness
    score = score * (0.9 + Math.random() * 0.2);
    
    return {
      name: model.name,
      score: parseFloat(score.toFixed(2)),
      costPerToken: model.costPerToken,
      qualityScore: model.qualityScore,
      latencyScore: parseFloat((0.5 + Math.random() * 0.5).toFixed(2))
    };
  }).sort((a, b) => b.score - a.score);
  
  // Create reason based on strategy
  let reason = '';
  if (strategy === 'cost') {
    reason = `Selected ${selectedModel} to optimize for lower token cost while maintaining acceptable quality`;
  } else if (strategy === 'quality') {
    reason = `Selected ${selectedModel} to maximize response quality for this complex query`;
  } else {
    reason = `Selected ${selectedModel} as the optimal balance between cost and quality for this query`;
  }
  
  return {
    timestamp: new Date().toISOString(),
    inputTokenEstimate: tokenEstimate,
    strategy,
    candidateModels,
    selectedModel,
    reason
  };
}

// Generate metadata for a model response
export function generateModelMetadata(
  modelId: string,
  promptLength: number,
  responseLength: number
): ModelMetadata {
  const modelInfo = MODELS[modelId as keyof typeof MODELS];
  const promptTokens = Math.floor(promptLength / 4) + 10;
  const completionTokens = Math.floor(responseLength / 4) + 20;
  const totalTokens = promptTokens + completionTokens;
  
  // Calculate cost
  const cost = totalTokens * (modelInfo?.costPerToken || 0.00001);
  
  // Generate reasonable latency based on model and tokens
  let baseLatency = 500;
  if (modelId === 'openai/gpt-4') baseLatency = 1800;
  else if (modelId === 'anthropic/claude-2') baseLatency = 1500;
  else if (modelId === 'meta-llama/llama-2-70b') baseLatency = 1200;
  
  const latencyMs = baseLatency + (totalTokens * 2) + (Math.random() * 500);
  
  return {
    model: modelId,
    latencyMs: Math.floor(latencyMs),
    promptTokens,
    completionTokens,
    totalTokens,
    cost,
    routingRationale: `Selected based on ${modelId} capabilities for this query type`
  };
}

// Generate comparison results for the compare page
export function generateComparisonData(prompt: string): ModelComparison[] {
  const modelsToCompare = ['openai/gpt-4', 'openai/gpt-3.5-turbo', 'anthropic/claude-2'];
  
  return modelsToCompare.map(modelId => {
    // Generate different responses for each model
    let response = '';
    if (modelId === 'openai/gpt-4') {
      response = `This is a comprehensive response from ${modelId} that demonstrates deep understanding and nuanced reasoning.\n\nThe output shows detailed analysis with multiple perspectives, well-structured argumentation, and accurate technical details where relevant.\n\nThe response quality justifies the higher cost and slightly increased latency compared to other models.`;
    } else if (modelId === 'openai/gpt-3.5-turbo') {
      response = `This is a good response from ${modelId} that covers the main points efficiently.\n\nThe output is correct and helpful, though less detailed than GPT-4's response.\n\nThis model provides an excellent balance of cost and quality for most general-purpose queries.`;
    } else {
      response = `This is an alternative perspective from ${modelId} with its own unique strengths.\n\nThe model shows particular capabilities in reasoning through problems step-by-step and handling certain specialized domains.\n\nThe response demonstrates how different architectures may have different performance characteristics on the same prompt.`;
    }
    
    return {
      model: modelId,
      response,
      metadata: generateModelMetadata(modelId, prompt.length, response.length)
    };
  });
}