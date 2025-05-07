import { FC } from 'react';
import ReactMarkdown from 'react-markdown';

interface MessageContentProps {
  content: string;
}

export const MessageContent: FC<MessageContentProps> = ({ content }) => {
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none">
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  );
};