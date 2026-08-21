# Local Multi-Agent AI Research Assistant

A local-first, autonomous multi-agent research assistant built with **Python**, **Ollama** (optimized for 3B open-weight models), a **Directed Acyclic Graph (DAG)** execution loop, and custom Python tool calling.

## 🏗️ Architecture & Execution Loop

### 🔄 Multi-Agent Directed Acyclic Graph (DAG)

```mermaid
flowchart TD
    %% User Entrypoint
    subgraph Input["👤 User Entrypoint"]
        Query(["User Query / Research Topic"])
    end

    %% Multi-Agent Directed Acyclic Graph Loop
    subgraph DAG["🔄 Autonomous Multi-Agent DAG Loop"]
        direction TB

        Planner["🧠 Planner Agent<br/>• Decomposes query into sub-queries<br/>• Targets knowledge gaps on re-planning"]

        Researcher["🔍 Research Agent<br/>• Orchestrates Python search tools<br/>• Scrapes token-bounded page extracts"]

        subgraph Tools["🛠️ Tool Suite"]
            DDG["DuckDuckGo Search API"]
            Wiki["Wikipedia Client"]
            Arxiv["arXiv Research Paper API"]
            Scraper["HTML/Text Web Scraper"]
        end

        State[("📚 Research State & Memory<br/>• Deduplicated Sources<br/>• Grounded Atomic Facts")]

        Critic["⚖️ Critic Agent<br/>• Scores Grounding & Relevance (0–100)<br/>• Identifies missing perspectives & gaps"]

        Decision{"📊 Quality Gate<br/>Confidence &ge; Threshold<br/>OR Max Iterations?"}

        GapAnalysis["💡 Gap Refinement<br/>Injects feedback & follow-up queries"]
    end

    %% Deliverables
    subgraph Output["📑 Output Deliverables"]
        Synthesizer["📝 Synthesizer Agent<br/>• Assembles cited research findings<br/>• Drafts Executive Summary & Sections"]
        Report[/"📄 Comprehensive Markdown Report<br/>&amp; Structured JSON State"/]
    end

    %% Flow Connections
    Query --> Planner
    Planner -->|Targeted Sub-Queries| Researcher
    Researcher --> Tools
    Tools --> Researcher
    Researcher -->|Extracted Evidence| State
    State --> Critic
    Critic -->|Quality Assessment & Score| Decision

    %% Conditional Routing
    Decision -->|❌ Insufficient<br/>Score &lt; Threshold| GapAnalysis
    GapAnalysis -->|Refined Objectives| Planner
    Decision -->|✅ Sufficient<br/>Score &ge; Threshold OR Max Iterations| Synthesizer

    Synthesizer --> Report

    %% Styling
    style Input fill:#0f172a,stroke:#38bdf8,stroke-width:1px,color:#f8fafc
    style DAG fill:#0b1120,stroke:#64748b,stroke-width:1px,color:#f8fafc
    style Tools fill:#1e1b4b,stroke:#818cf8,stroke-width:1px,color:#f8fafc
    style Output fill:#064e3b,stroke:#34d399,stroke-width:1px,color:#f8fafc

    style Planner fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    style Researcher fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    style Critic fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    style Synthesizer fill:#1e293b,stroke:#34d399,stroke-width:2px,color:#f8fafc
    style Decision fill:#312e81,stroke:#a855f7,stroke-width:2px,color:#f8fafc
    style GapAnalysis fill:#451a03,stroke:#f59e0b,stroke-width:1.5px,color:#fef3c7
    style State fill:#14532d,stroke:#22c55e,stroke-width:1.5px,color:#f0fdf4
    style Query fill:#1e293b,stroke:#94a3b8,stroke-width:1.5px,color:#f8fafc
    style Report fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style DDG fill:#312e81,stroke:#6366f1,stroke-width:1px,color:#e0e7ff
    style Wiki fill:#312e81,stroke:#6366f1,stroke-width:1px,color:#e0e7ff
    style Arxiv fill:#312e81,stroke:#6366f1,stroke-width:1px,color:#e0e7ff
    style Scraper fill:#312e81,stroke:#6366f1,stroke-width:1px,color:#e0e7ff
```

### 🔁 Execution Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as 👤 User / Web UI
    participant DAG as ⚙️ DAG Engine
    participant Planner as 🧠 Planner Agent
    participant Researcher as 🔍 Research Agent
    participant Tools as 🛠️ Python Tools
    participant Critic as ⚖️ Critic Agent
    participant Synthesizer as 📝 Synthesizer Agent

    User->>DAG: Execute Research Query (Topic, Model, Threshold)
    
    loop Dynamic Research Loop (Until Confident OR Max Iterations)
        DAG->>Planner: Request Sub-Queries & Strategy
        Planner-->>DAG: PlanOutput (2–4 Targeted Queries)
        
        DAG->>Researcher: Dispatch Search Sub-Queries
        loop For Each Query
            Researcher->>Tools: Query DDG, Wikipedia, arXiv & Web Scraper
            Tools-->>Researcher: Return Documents & Scraped Text
            Researcher->>Researcher: Extract Atomic Facts & Deduplicate
        end
        Researcher-->>DAG: Updated State (Sources + Facts)
        
        DAG->>Critic: Evaluate Collected Evidence
        Critic-->>DAG: CritiqueOutput (Score, Grounding, Gaps)
        
        alt Score >= Threshold OR Iterations >= Max
            Note over DAG: Quality threshold met — proceed to synthesis
        else Score < Threshold (Re-planning needed)
            Note over DAG: Inject identified gaps into next iteration plan
        end
    end
    
    DAG->>Synthesizer: Compile Final Grounded Report
    Synthesizer-->>DAG: SynthesisOutput (Summary, Findings, Citations)
    DAG->>User: Deliver Publication-Ready Markdown Report & JSON State
```

---

## 🚀 Interfaces & User Experience

### 1. Interactive Streamlit Web Dashboard (`app.py`)
A full web UI featuring real-time DAG progress tracking, interactive agent tabs, and one-click file downloads.

```bash
# Launch Streamlit Web UI
streamlit run app.py
```

**Web UI Features:**
- ⚙️ **Sidebar Controls**: Model selection, Ollama endpoint detector, mock simulator toggle, confidence threshold slider, max iterations, search provider toggles.
- 📌 **Preset Topics**: 1-click test runs on complex research domains.
- 📑 **Report Viewer**: Rendered Markdown with citation links, executive summaries, and `.md`/`.json` download buttons.
- 🧠 **Planner Analysis**: Sub-query decomposition inspection and reasoning.
- 🌐 **Evidence & Fact Explorer**: Expandable source document previews and grounded atomic facts.
- ⚖️ **Critic Quality Audit**: Confidence, relevance, and factual grounding meters + auditor feedback.
- ⏱️ **DAG Execution Trace**: Interactive step-by-step audit trail table.

---

### 2. Enhanced Terminal Rich CLI (`local_researcher/cli.py`)

```bash
# Run with Local Ollama
python -m local_researcher.cli "Autonomous Multi-Agent AI Architectures in 2025" --model llama3.2:3b --output-file report.md

# Run Offline / Mock Mode
python -m local_researcher.cli "Deep Learning Optimization Techniques" --mock --output-file report.md --json-out state.json
```

**CLI Features:**
- 🎨 Modern color-coded agent tags (`[PLANNER]`, `[RESEARCHER]`, `[CRITIC]`, `[SYNTHESIZER]`).
- 📊 Real-time ASCII progress gauges (`[████████░░] 85%`).
- ⚖️ Critic scorecard table and decision gate status.
- 🏆 Post-execution summary card with elapsed timing, iterations count, and source metrics.

---

## 🧪 Testing

```bash
# Run the complete test suite (27 unit & integration tests)
python -m pytest -v
```
