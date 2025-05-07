'use client';

import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Code2 } from 'lucide-react';

interface DeveloperModeToggleProps {
  value: boolean;
  onChange: (value: boolean) => void;
}

export function DeveloperModeToggle({ value, onChange }: DeveloperModeToggleProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center space-x-2">
        <Code2 className="h-4 w-4 text-muted-foreground" />
        <Label htmlFor="developer-mode" className="flex-1">
          Developer Mode
        </Label>
        <Switch
          id="developer-mode"
          checked={value}
          onCheckedChange={onChange}
        />
      </div>
      <p className="text-xs text-muted-foreground">
        Show complete routing trace and technical details for debugging
      </p>
    </div>
  );
}