# AGENTS.md — Review Integrity Bot

This repository hosts a standalone Day 35 AI application for AI Advent Challenge #8.

## Mission

Build an honest-review analysis service that:
- ingests reviews from adapters (starting with fixture and Google-focused source),
- classifies review quality with LLM structured output,
- recomputes a quality-aware true rating for transparent comparison.

## Non-Goals

- No coupling to LocalStack core codebase.
- No hidden business logic in HTTP handlers.
- No secrets committed to git.

## Engineering Principles

- Async-first Python service design.
- Strict Pydantic contracts at API and LLM boundaries.
- Layered architecture with adapter-driven external integrations.
- Reproducible local demo path from fixture data.
