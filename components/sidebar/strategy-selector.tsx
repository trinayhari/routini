'use client';

import { RoutingStrategy } from '@/types';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Sparkles, Coins, Scale } from 'lucide-react';

interface StrategySelectorProps {
  value: RoutingStrategy;
  onChange: (value: RoutingStrategy) => void;
}

export function StrategySelector({ value, onChange }: StrategySelectorProps) {
  return (
    <div className="space-y-2">
      <Label htmlFor="strategy">Routing Strategy</Label>
      <Select value={value} onValueChange={(val) => onChange(val as RoutingStrategy)}>
        <SelectTrigger id="strategy" className="w-full">
          <SelectValue placeholder="Select a strategy" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="cost" className="flex items-center">
            <div className="flex items-center gap-2">
              <Coins className="h-4 w-4 text-amber-500" />
              <span>Cost-optimized</span>
            </div>
          </SelectItem>
          <SelectItem value="quality">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-violet-500" />
              <span>Quality-optimized</span>
            </div>
          </SelectItem>
          <SelectItem value="balanced">
            <div className="flex items-center gap-2">
              <Scale className="h-4 w-4 text-teal-500" />
              <span>Balanced</span>
            </div>
          </SelectItem>
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">
        {value === 'cost'
          ? 'Optimizes for lower token costs using efficient models'
          : value === 'quality'
          ? 'Prioritizes response quality using advanced models'
          : 'Balances cost and quality for optimal results'}
      </p>
    </div>
  );
}