'use client';

import { useState } from 'react';
import { ModelComparison } from '@/types';
import { ComparisonCard } from '@/components/compare/comparison-card';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { compareModels } from '@/lib/api';
import { generateComparisonData } from '@/lib/mock-data';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Loader2 } from 'lucide-react';

export function ComparisonView() {
  const [prompt, setPrompt] = useState('');
  const [results, setResults] = useState<ModelComparison[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  
  const handleCompare = async () => {
    if (!prompt.trim()) return;
    
    setIsLoading(true);
    
    try {
      // Call the API to get comparison results
      const response = await compareModels({
        prompt,
        models: ['gpt-4', 'gpt-3.5-turbo', 'claude-2']
      });
      
      setResults(response.results);
    } catch (error) {
      console.error('Error comparing models:', error);
      // In case of error, use mock data
      setResults(generateComparisonData(prompt));
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b">
        <h2 className="text-xl font-semibold mb-4">Model Comparison</h2>
        <div className="space-y-4">
          <Textarea
            placeholder="Enter your prompt to compare across models..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="min-h-[120px] resize-none"
            disabled={isLoading}
          />
          <Button 
            onClick={handleCompare} 
            disabled={!prompt.trim() || isLoading}
            className="w-full"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Comparing models...
              </>
            ) : (
              'Compare Models'
            )}
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {results.length > 0 ? (
          <div className="space-y-4">
            <Tabs defaultValue="side-by-side" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="side-by-side">Side by Side</TabsTrigger>
                <TabsTrigger value="tabbed">Tabbed View</TabsTrigger>
              </TabsList>
              
              <TabsContent value="side-by-side" className="mt-4">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {results.map((result) => (
                    <ComparisonCard key={result.model} result={result} />
                  ))}
                </div>
              </TabsContent>
              
              <TabsContent value="tabbed" className="mt-4">
                <Tabs defaultValue={results[0].model} className="w-full">
                  <TabsList className="grid w-full grid-cols-3">
                    {results.map((result) => (
                      <TabsTrigger key={result.model} value={result.model}>
                        {result.model}
                      </TabsTrigger>
                    ))}
                  </TabsList>
                  
                  {results.map((result) => (
                    <TabsContent key={result.model} value={result.model} className="mt-4">
                      <ComparisonCard result={result} fullWidth />
                    </TabsContent>
                  ))}
                </Tabs>
              </TabsContent>
            </Tabs>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            {isLoading ? (
              <div className="flex flex-col items-center">
                <Loader2 className="h-8 w-8 animate-spin mb-2" />
                <p>Comparing models...</p>
              </div>
            ) : (
              <p>Enter a prompt and click "Compare Models" to see results</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}