# Response Cache Directory

This directory contains cached responses from LLM API calls, identified by SHA-256 hashes of prompts.

## Overview

The caching system stores responses to avoid redundant API calls to language models, which:

- Reduces costs
- Improves response times
- Decreases load on LLM providers

## How it Works

1. When a prompt is received, a SHA-256 hash is generated based on:

   - The prompt text
   - Temperature and other generation parameters
   - Previous messages (if in a conversation context)

2. If a matching hash is found in the cache, the cached response is returned immediately
3. Otherwise, the LLM API is called and the response is cached for future use

## Cache Files

Each cache file is named with the SHA-256 hash of the prompt and has the `.json` extension.

Cache files contain:

- The full response from the LLM
- Metadata such as tokens used, cost, and latency
- A timestamp for expiration tracking

## Maintenance

The cache is automatically maintained with these processes:

- Expired entries are cleared on application startup
- Entries expire after 24 hours by default (configurable via CACHE_EXPIRATION env var)
- The `/generate/cache` endpoint can be used to view cache statistics
- The `/generate/cache` DELETE endpoint can be used to manually clear expired cache entries

## Configuration

You can configure the cache with these environment variables:

- `CACHE_DIR`: Directory for cache files (default: "cache")
- `CACHE_EXPIRATION`: Cache expiration time in seconds (default: 86400 = 24 hours)
