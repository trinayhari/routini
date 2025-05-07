#!/usr/bin/env node
/**
 * Test script for Next.js API routes
 * 
 * This script tests the API routes in the Next.js application
 * to ensure they are correctly communicating with the FastAPI backend.
 */

const fetch = require('node-fetch');

// Configuration
const config = {
  nextjsUrl: process.env.NEXTJS_URL || 'http://localhost:3000',
  timeout: 30000  // 30 seconds
};

// Helper function for colored console output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  white: '\x1b[37m'
};

function print(color, message) {
  console.log(`${color}${message}${colors.reset}`);
}

// Test models
const TEST_MODELS = [
  'openai/gpt-4',
  'anthropic/claude-3-opus',
  'mistralai/mixtral-8x7b-instruct'
];

/**
 * Test the /api/generate endpoint
 */
async function testGenerateEndpoint() {
  print(colors.blue, '\n=== Testing /api/generate endpoint ===');
  
  const testCases = [
    {
      name: 'Basic question with balanced strategy',
      payload: {
        prompt: 'What is the capital of France?',
        strategy: 'balanced'
      },
      expectedContains: 'paris'
    },
    {
      name: 'Code request with quality strategy',
      payload: {
        prompt: 'Write a simple JavaScript function to calculate fibonacci numbers',
        strategy: 'quality'
      },
      expectedContains: 'function fibonacci'
    }
  ];
  
  for (const test of testCases) {
    print(colors.cyan, `\nTest: ${test.name}`);
    
    try {
      const startTime = Date.now();
      
      const response = await fetch(`${config.nextjsUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(test.payload),
        timeout: config.timeout
      });
      
      const elapsed = (Date.now() - startTime) / 1000;
      print(colors.white, `  Response time: ${elapsed.toFixed(2)}s`);
      
      if (response.ok) {
        const data = await response.json();
        
        if (data.error) {
          print(colors.red, `  ❌ Error: ${data.error} - ${data.details || ''}`);
          continue;
        }
        
        print(colors.green, `  ✅ Success!`);
        print(colors.white, `  Model: ${data.message.metadata.model}`);
        print(colors.white, `  Latency: ${(data.message.metadata.latencyMs / 1000).toFixed(2)}s`);
        print(colors.white, `  Tokens: ${data.message.metadata.totalTokens} | Cost: $${data.message.metadata.cost.toFixed(6)}`);
        
        // Check response content
        const responseText = data.message.content.toLowerCase();
        const expected = test.expectedContains.toLowerCase();
        
        if (responseText.includes(expected)) {
          print(colors.green, `  ✅ Contains expected text: '${expected}'`);
        } else {
          print(colors.red, `  ❌ Missing expected text: '${expected}'`);
          print(colors.white, `  Response: ${data.message.content.slice(0, 100)}...`);
        }
      } else {
        let errorText = '';
        try {
          const errorData = await response.json();
          errorText = JSON.stringify(errorData);
        } catch (e) {
          errorText = await response.text();
        }
        
        print(colors.red, `  ❌ HTTP Error ${response.status}: ${errorText}`);
      }
    } catch (error) {
      print(colors.red, `  ❌ Exception: ${error.message}`);
    }
  }
}

/**
 * Test the /api/compare endpoint
 */
async function testCompareEndpoint() {
  print(colors.blue, '\n=== Testing /api/compare endpoint ===');
  print(colors.cyan, '\nTest: Model comparison for "What is the capital of France?"');
  
  try {
    const startTime = Date.now();
    
    const response = await fetch(`${config.nextjsUrl}/api/compare`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        prompt: 'What is the capital of France?',
        models: TEST_MODELS
      }),
      timeout: config.timeout
    });
    
    const elapsed = (Date.now() - startTime) / 1000;
    print(colors.white, `  Response time: ${elapsed.toFixed(2)}s`);
    
    if (response.ok) {
      const data = await response.json();
      
      if (data.error) {
        print(colors.red, `  ❌ Error: ${data.error} - ${data.details || ''}`);
        return;
      }
      
      print(colors.green, `  ✅ Success! Compared ${data.results.length} models:`);
      
      // Verify each model response
      for (const result of data.results) {
        print(colors.cyan, `\n  Model: ${result.model}`);
        print(colors.white, `  Latency: ${(result.metadata.latencyMs / 1000).toFixed(2)}s | Cost: $${result.metadata.cost.toFixed(6)} | Tokens: ${result.metadata.totalTokens}`);
        
        const responseText = result.response.toLowerCase();
        if (responseText.includes('paris')) {
          print(colors.green, `  ✅ Contains "Paris"`);
        } else {
          print(colors.red, `  ❌ "Paris" not found in response`);
          print(colors.white, `  Response: ${result.response.slice(0, 100)}...`);
        }
      }
    } else {
      let errorText = '';
      try {
        const errorData = await response.json();
        errorText = JSON.stringify(errorData);
      } catch (e) {
        errorText = await response.text();
      }
      
      print(colors.red, `  ❌ HTTP Error ${response.status}: ${errorText}`);
    }
  } catch (error) {
    print(colors.red, `  ❌ Exception: ${error.message}`);
  }
}

/**
 * Main function to run all tests
 */
async function runTests() {
  print(colors.magenta, `\nTesting Next.js API routes at ${config.nextjsUrl}`);
  
  // Test the generate endpoint
  await testGenerateEndpoint();
  
  // Test the compare endpoint
  await testCompareEndpoint();
  
  print(colors.magenta, '\nAll tests completed!');
}

// Run the tests
runTests().catch(error => {
  console.error('Failed to run tests:', error);
  process.exit(1);
}); 