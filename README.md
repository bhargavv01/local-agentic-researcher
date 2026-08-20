# Local Multi-Agent AI Research Assistant

A local-first, autonomous multi-agent research assistant built with **Python**, **Ollama** (optimized for 3B open-weight models), a **Directed Acyclic Graph (DAG)** execution loop, and custom Python tool calling.

---

## Architecture & System Overview

```
User Query
    │
    ▼
┌────────────────────────────────────────────────────────┐
│                      PLANNER AGENT                     │
│  Decomposes query into targeted search sub-queries      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                     RESEARCH AGENT                     │
│  Executes Python tools (DuckDuckGo, Wikipedia, arXiv)  │
│  Scrapes & extracts clean, token-bounded snippets      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                      CRITIC AGENT                      │
│  Evaluates factual grounding, relevance & coverage     │
│  Assigns confidence score (0 - 100)                     │
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │ Score >= 75 OR Iter >= 3? │
              └─────────────┬─────────────┘
                     No     │     Yes
         ┌──────────────────┘     └──────────────────┐
         ▼                                           ▼
┌─────────────────┐                       ┌─────────────────────┐
│  RE-PLAN / GAP  │                       │  SYNTHESIZER AGENT  │
│   REFINEMENT    │                       │  Produces Cited     │
│  (Loop to Step) │                       │  Report (MD & JSON) │
└─────────────────┘                       └─────────────────────┘
```

---

## Phase 1 Completed Components

1. **State & Data Contracts (`local_researcher.models.state`)**:
   - Strongly-typed Pydantic V2 schemas for global DAG state (`ResearchState`), steps, source documents, atomic facts, and agent I/O contracts (`PlanOutput`, `CritiqueOutput`, `SynthesisOutput`).
2. **Small LLM (3B) Client Engine (`local_researcher.llm.client`)**:
   - Ollama client with JSON schema constraints, automatic markdown/JSON fence stripping, trailing comma repair, and multi-retry error reflection.
   - Deterministic `MockLLMClient` for offline execution, unit testing, and benchmarking without an active GPU.
3. **Custom Python Tool Calling (`local_researcher.tools`)**:
   - `DuckDuckGoSearchEngine`: Live web search with direct library and HTML fallback.
   - `WikipediaSearchEngine`: Encyclopedic queries via official Wikipedia APIs.
   - `ArxivSearchEngine`: Scholarly search via arXiv XML/Atom API.
   - `WebScraper`: Text extraction with `trafilatura` and `BeautifulSoup4` with context-window token bounding.
4. **Test Suite (`tests/`)**:
   - 17 comprehensive unit tests covering models, JSON schema parsing, tool extraction, and LLM repair logic.

---

## Quickstart & Testing

```bash
# Run test suite
python -m pytest -v
```
