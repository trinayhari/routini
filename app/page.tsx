'use client';

import { ChatInterface } from '@/components/chat/chat-interface';
import { MainLayout } from '@/components/layout/main-layout';
import { RoutingStrategy } from '@/types';

interface HomePageProps {
  strategy?: RoutingStrategy;
  developerMode?: boolean;
}

export default function Home({ 
  strategy = 'balanced',
  developerMode = false 
}: HomePageProps) {
  return (
    <MainLayout>
      <ChatInterface strategy={strategy} developerMode={developerMode} />
    </MainLayout>
  );
}