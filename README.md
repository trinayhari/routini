# BestRoute - AI Model Router

A Next.js application that intelligently routes prompts to the most suitable AI model based on cost, quality, and latency.

## Prerequisites

- Node.js 18.x or later
- npm or yarn
- OpenAI API key
- Anthropic API key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/bestRoute.git
cd bestRoute
```

2. Install dependencies:

```bash
npm install
# or
yarn install
```

3. Create a `.env.local` file in the root directory with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Running the Application

1. Start the development server:

```bash
npm run dev
# or
yarn dev
```

2. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Features

- Intelligent model routing based on:
  - Cost optimization
  - Quality optimization
  - Balanced approach
- Real-time cost and token tracking
- Developer mode with detailed routing traces
- Chat history with model selection explanations

## Tech Stack

- Next.js 14
- TypeScript
- Tailwind CSS
- Shadcn/ui
- OpenAI API
- Anthropic API

## Project Structure

```
bestRoute/
├── app/              # Next.js app directory
├── components/       # React components
├── lib/             # Utility functions
├── types/           # TypeScript types
└── public/          # Static assets
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
