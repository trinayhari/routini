# Frontend Integration with FastAPI Backend

This guide explains how to connect the Next.js frontend with the FastAPI backend for model routing.

## Setup

1. Start the FastAPI backend:

   ```bash
   cd backend
   python run.py
   ```

   The server will run on http://localhost:8000

2. Create an `.env.local` file in the project root with:

   ```
   NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
   ```

3. Start the Next.js frontend:
   ```bash
   npm run dev
   ```
   The frontend will run on http://localhost:3000

## Integration Details

### API Layer

The frontend integrates with the backend through two layers:

1. **Direct Backend Connection**: `lib/api.ts` contains the `sendPrompt()` function that directly communicates with the FastAPI backend.

2. **Next.js API Routes**: `/app/api/generate/route.ts` acts as a proxy between the frontend and backend, transforming data formats as needed.

### Usage in Components

To use the backend in a React component:

```tsx
import { generateResponse } from "@/lib/api";
import { useState } from "react";

export default function ChatComponent() {
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(prompt: string) {
    setLoading(true);
    try {
      const result = await generateResponse({
        prompt,
        strategy: "balanced",
      });
      setResponse(result.message.content);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  }

  return <div>{/* UI components */}</div>;
}
```

## Mapping between Frontend and Backend

| Frontend Strategy | Backend Strategy |
| ----------------- | ---------------- |
| 'cost'            | 'cheapest'       |
| 'quality'         | 'most_capable'   |
| 'balanced'        | 'balanced'       |

## Response Format

The backend returns:

- Model used
- Generated text
- Latency in seconds
- Estimated cost
- Routing explanation

The frontend transforms this into the appropriate format for the UI.
