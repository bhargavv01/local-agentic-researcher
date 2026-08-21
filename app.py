"""
Interactive Streamlit Web Dashboard for Local Multi-Agent AI Research Assistant.
Provides a modern UI for real-time DAG execution, agent step inspection,
confidence telemetry, and report exports.
"""

import json
import time
from datetime import datetime
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="Multi-Agent AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .badge-planner { background-color: #1E40AF; color: #DBEAFE; padding: 3px 8px; border-radius: 6px; font-size: 0.8rem; }
    .badge-researcher { background-color: #065F46; color: #D1FAE5; padding: 3px 8px; border-radius: 6px; font-size: 0.8rem; }
    .badge-critic { background-color: #92400E; color: #FEF3C7; padding: 3px 8px; border-radius: 6px; font-size: 0.8rem; }
    .badge-synthesizer { background-color: #5B21B6; color: #EDE9FE; padding: 3px 8px; border-radius: 6px; font-size: 0.8rem; }
</style>
""",
    unsafe_allow_html=True,
)

from local_researcher.graph.dag import ResearchGraph
from local_researcher.graph.events import GraphEvent, GraphEventType
from local_researcher.llm.client import OllamaClient, get_llm_client
from local_researcher.models.state import ResearchState
from local_researcher.tools.search import UnifiedSearchTool

# ==========================================
# Sidebar Configuration
# ==========================================
st.sidebar.markdown("### ⚙️ Workflow Settings")

# Check Ollama status
default_ollama = OllamaClient(model_name="llama3.2:3b")
ollama_online = default_ollama.is_available()

if ollama_online:
    st.sidebar.success("🟢 Local Ollama Server Detected")
    available_models = default_ollama.list_models()
    model_options = available_models if available_models else ["llama3.2:3b", "qwen2.5:3b", "mistral:7b"]
else:
    st.sidebar.info("🟡 Ollama Offline (Mock Mode Available)")
    model_options = ["llama3.2:3b", "qwen2.5:3b", "phi3.5:3.8b", "mistral:7b"]

selected_model = st.sidebar.selectbox("LLM Model (Ollama)", options=model_options, index=0)
force_mock = st.sidebar.toggle("Force Offline Mock Mode", value=(not ollama_online))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 DAG Control Parameters")
confidence_threshold = st.sidebar.slider("Critic Confidence Gate (%)", min_value=50, max_value=95, value=75, step=5)
max_iterations = st.sidebar.slider("Max Feedback Iterations", min_value=1, max_value=5, value=3, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌐 Tool Integration")
enable_web = st.sidebar.checkbox("DuckDuckGo Web Search", value=True)
enable_wiki = st.sidebar.checkbox("Wikipedia API", value=True)
enable_arxiv = st.sidebar.checkbox("arXiv Scholarly API", value=True)

# ==========================================
# Main Header & Prompting
# ==========================================
st.markdown('<div class="main-header">🧠 Multi-Agent AI Research Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Local-first autonomous research loop with Planner, Researcher, Critic, and Synthesizer agents.</div>',
    unsafe_allow_html=True,
)

# Preset Query Buttons
col_p1, col_p2, col_p3 = st.columns(3)
preset_query = None
if col_p1.button("📌 Autonomous Multi-Agent AI (2025)", use_container_width=True):
    preset_query = "Autonomous Multi-Agent AI Architectures and Benchmark Evaluations in 2025"
if col_p2.button("📌 Quantum Key Distribution Protocols", use_container_width=True):
    preset_query = "Quantum Key Distribution protocols and vulnerability analysis"
if col_p3.button("📌 Small Language Models (3B) Optimization", use_container_width=True):
    preset_query = "Optimization techniques and structured JSON generation for 3B parameter LLMs"

# Query Input
user_query = st.text_area(
    "Research Query or Topic:",
    value=preset_query if preset_query else "Explain DAG execution loops and critic confidence gates in AI research agents",
    height=80,
    placeholder="Enter the technical topic or research question you want to investigate...",
)

run_button = st.button("🚀 Start Autonomous Research", type="primary", use_container_width=True)

# Initialize Session State for results
if "research_state" not in st.session_state:
    st.session_state["research_state"] = None
if "logs" not in st.session_state:
    st.session_state["logs"] = []

# ==========================================
# Research Execution Handling
# ==========================================
if run_button and user_query.strip():
    progress_container = st.container()
    status_text = progress_container.empty()
    progress_bar = progress_container.progress(0.0)

    events_log = []

    def ui_event_callback(event: GraphEvent):
        events_log.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "node": event.node_name,
                "type": str(event.event_type.value if hasattr(event.event_type, "value") else event.event_type),
                "message": event.message,
            }
        )
        status_text.markdown(f"**Current Action:** `{event.node_name.upper()}` — {event.message}")

    with st.spinner("Executing Autonomous Research DAG Loop..."):
        llm_client = get_llm_client(model_name=selected_model, force_mock=force_mock)
        search_tool = UnifiedSearchTool(
            enable_web=enable_web,
            enable_wikipedia=enable_wiki,
            enable_arxiv=enable_arxiv,
        )

        graph = ResearchGraph(llm_client=llm_client, search_tool=search_tool)
        graph.add_callback(ui_event_callback)

        start_time = time.time()
        try:
            state: ResearchState = graph.run(
                query=user_query.strip(),
                max_iterations=max_iterations,
                confidence_threshold=confidence_threshold,
            )
            elapsed = time.time() - start_time
            progress_bar.progress(1.0)
            status_text.success(f"✨ Research completed successfully in {elapsed:.2f}s!")
            st.session_state["research_state"] = state
            st.session_state["logs"] = events_log
            st.session_state["elapsed"] = elapsed
        except Exception as e:
            status_text.error(f"❌ Execution failed: {e}")

# ==========================================
# Display Research Results & Tabs
# ==========================================
state: ResearchState | None = st.session_state.get("research_state")

if state and state.status == "completed":
    st.markdown("---")
    
    # Key Performance Metrics
    final_conf = (
        state.final_synthesis.final_confidence_score
        if state.final_synthesis
        else (state.latest_critique.confidence_score if state.latest_critique else 0)
    )
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Confidence Score", f"{final_conf}%", delta=f"{final_conf - confidence_threshold}% vs Target")
    m2.metric("Feedback Iterations", f"{state.iteration + 1} of {max_iterations}")
    m3.metric("Verified Sources", f"{len(state.collected_sources)}")
    m4.metric("Extracted Facts", f"{len(state.extracted_facts)}")

    # Main Tabs
    tab_report, tab_planner, tab_sources, tab_critic, tab_trace = st.tabs(
        [
            "📄 Research Report",
            "🧠 Planner Breakdown",
            "🌐 Sources & Evidence",
            "⚖️ Critic Scorecard",
            "⏱️ Execution Trace",
        ]
    )

    # 1. Final Research Report Tab
    with tab_report:
        if state.final_report_markdown:
            st.markdown(state.final_report_markdown)
            
            st.markdown("---")
            d_col1, d_col2 = st.columns(2)
            d_col1.download_button(
                label="📥 Download Markdown Report (.md)",
                data=state.final_report_markdown,
                file_name=f"research_report_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True,
            )
            d_col2.download_button(
                label="📥 Download State & Trace (.json)",
                data=state.model_dump_json(indent=2),
                file_name=f"research_state_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True,
            )

    # 2. Planner Tab
    with tab_planner:
        if state.current_plan:
            st.markdown("### 🎯 Sub-Query Decomposition")
            st.info(f"**Strategic Reasoning:** {state.current_plan.reasoning}")
            
            st.markdown("#### Generated Search Queries")
            for idx, q in enumerate(state.current_plan.search_queries, 1):
                st.markdown(f"- **Query {idx}:** `{q}`")
                
            if state.current_plan.focus_aspects:
                st.markdown("#### Core Focus Angles")
                for fa in state.current_plan.focus_aspects:
                    st.markdown(f"- 🔹 {fa}")

    # 3. Sources & Grounded Facts Tab
    with tab_sources:
        st.markdown(f"### 🌐 Collected Evidence ({len(state.collected_sources)} Sources)")
        
        for idx, src in enumerate(state.collected_sources, 1):
            with st.expander(f"Source {idx}: {src.title} ({src.source_type.upper()})"):
                st.markdown(f"**URL / Identifier:** [{src.url}]({src.url})")
                st.markdown(f"**Relevance Score:** `{src.relevance_score}`")
                st.markdown(f"**Extracted Content Snippet:**\n> {src.snippet}")

        st.markdown(f"### 💡 Grounded Atomic Facts ({len(state.extracted_facts)})")
        for fact in state.extracted_facts:
            st.markdown(f"- ✅ **{fact.statement}** *(Source: [{fact.source_title}]({fact.source_url}))*")

    # 4. Critic Quality Scorecard Tab
    with tab_critic:
        if state.latest_critique:
            c = state.latest_critique
            st.markdown("### ⚖️ Auditor Evaluation")
            
            c_col1, c_col2, c_col3 = st.columns(3)
            c_col1.metric("Overall Confidence", f"{c.confidence_score}%")
            c_col2.metric("Relevance Alignment", f"{c.relevance_score}%")
            c_col3.metric("Factual Grounding", f"{c.factual_grounding_score}%")

            if c.is_sufficient:
                st.success(f"**Quality Gate Status:** PASSED (Meets {confidence_threshold}% threshold)")
            else:
                st.warning(f"**Quality Gate Status:** REFINEMENT REQUIRED (Below {confidence_threshold}% threshold)")

            st.markdown(f"**Critic Assessment Feedback:**\n> {c.feedback}")

            if c.identified_gaps:
                st.markdown("#### ⚠️ Identified Knowledge Gaps")
                for gap in c.identified_gaps:
                    st.markdown(f"- 🔸 {gap}")

            if c.suggested_follow_up_queries:
                st.markdown("#### 🔍 Suggested Refinement Queries")
                for fq in c.suggested_follow_up_queries:
                    st.markdown(f"- 🔹 `{fq}`")

    # 5. Execution Timeline Tab
    with tab_trace:
        st.markdown("### ⏱️ DAG Step-by-Step Audit Trail")
        if state.execution_log:
            trace_data = [
                {
                    "Step ID": s.step_id,
                    "Agent": s.agent_name,
                    "Action": s.action,
                    "Status": s.status,
                    "Timestamp": s.timestamp,
                }
                for s in state.execution_log
            ]
            st.dataframe(trace_data, use_container_width=True)
