"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "@/components/theme-toggle";
import { Button } from "@/components/ui/button";
import { MessageSquare, GitCompareArrows, Beaker } from "lucide-react";

export function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-14 items-center px-4 md:px-6">
        <div className="flex flex-1 items-center justify-between">
          <nav className="flex items-center space-x-2">
            <Link href="/" className="hidden items-center space-x-2 md:flex">
              <span className="hidden text-xl font-bold sm:inline-block">
                LLM Router
              </span>
            </Link>
            <div className="flex items-center space-x-1">
              <Link href="/">
                <Button
                  variant={pathname === "/" ? "default" : "ghost"}
                  size="sm"
                  className="gap-1"
                >
                  <MessageSquare className="h-4 w-4" />
                  <span>Chat</span>
                </Button>
              </Link>
              <Link href="/compare">
                <Button
                  variant={pathname === "/compare" ? "default" : "ghost"}
                  size="sm"
                  className="gap-1"
                >
                  <GitCompareArrows className="h-4 w-4" />
                  <span>Compare</span>
                </Button>
              </Link>
              <Link href="/test">
                <Button
                  variant={pathname === "/test" ? "default" : "ghost"}
                  size="sm"
                  className="gap-1"
                >
                  <Beaker className="h-4 w-4" />
                  <span>Test</span>
                </Button>
              </Link>
            </div>
          </nav>
          <div className="flex items-center space-x-2">
            <ThemeToggle />
          </div>
        </div>
      </div>
    </header>
  );
}
