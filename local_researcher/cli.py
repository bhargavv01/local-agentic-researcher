"""
Interactive Rich Command-Line Interface for Local Multi-Agent Research Assistant.
Provides real-time visualization of DAG node execution, confidence gauges,
styled scorecard tables, and Markdown / JSON report export.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure UTF-8 output encoding across Windows terminals
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from local_researcher.graph.dag import ResearchGraph
from local_researcher.graph.events import GraphEvent, GraphEventType
from local_researcher.llm.client import get_llm_client
from local_researcher.models.state import ResearchState

console = Console()


def render_progress_bar(score: int, max_val: int = 100, width: int = 20) -> str:
    """Renders a colorful ASCII progress bar."""
    filled = int((score / max_val) * width)
    unfilled = width - filled
    bar = "█" * filled + "░" * unfilled
    color = "green" if score >= 75 else "yellow" if score >= 50 else "red"
    return f"[{color}]{bar} {score}%[/{color}]"


def create_event_listener(console: Console, max_iterations: int, threshold: int):
    """
    Creates an event callback handler to render DAG node events with rich styling in real-time.
    """

    def handle_event(event: GraphEvent):
        etype = event.event_type
        data = event.data

        if etype == GraphEventType.ITERATION_STARTED:
            console.print()
            console.print(
                Rule(
                    f"[bold cyan]🔄 DAG Feedback Loop — Iteration {event.iteration} of {max_iterations}[/bold cyan]",
                    style="cyan",
                )
            )

        elif etype == GraphEventType.NODE_STARTED:
            node_icons = {
                "planner": "[bold blue]🧠 [PLANNER][/bold blue]",
                "researcher": "[bold green]🔍 [RESEARCHER][/bold green]",
                "critic": "[bold yellow]⚖️ [CRITIC][/bold yellow]",
                "synthesizer": "[bold magenta]📝 [SYNTHESIZER][/bold magenta]",
            }
            label = node_icons.get(event.node_name, f"[bold][{event.node_name.upper()}][/bold]")
            console.print(f"  {label} {event.message}...")

        elif etype == GraphEventType.NODE_COMPLETED:
            if event.node_name == "planner" and "search_queries" in data:
                queries = data.get("search_queries", [])
                table = Table(
                    title="🎯 Generated Search Angles & Sub-Queries",
                    show_header=True,
                    header_style="bold blue",
                    border_style="blue",
                )
                table.add_column("#", style="dim", width=4, justify="center")
                table.add_column("Query", style="cyan")
                for idx, q in enumerate(queries, start=1):
                    table.add_row(str(idx), q)
                console.print(table)

            elif event.node_name == "researcher" and "total_sources" in data:
                total_sources = data.get("total_sources", 0)
                total_facts = data.get("total_facts", 0)
                console.print(
                    f"    [green]✨ Research aggregated:[/green] [bold cyan]{total_sources}[/bold cyan] sources processed, "
                    f"[bold green]{total_facts}[/bold green] grounded atomic facts extracted."
                )

        elif etype == GraphEventType.CRITIQUE_EVALUATED:
            score = data.get("confidence_score", 0)
            rel_score = data.get("relevance_score", 0)
            ground_score = data.get("factual_grounding_score", 0)
            is_suff = data.get("is_sufficient", False)
            feedback = data.get("feedback", "")
            gaps = data.get("identified_gaps", [])

            table = Table(
                title="⚖️ Critic Quality & Grounding Scorecard",
                show_header=True,
                header_style="bold yellow",
                border_style="yellow",
            )
            table.add_column("Evaluation Dimension", style="bold")
            table.add_column("Score / Visual Gauge", justify="left")
            table.add_column("Assessment Criteria", style="italic")

            table.add_row(
                "Overall Confidence",
                render_progress_bar(score),
                f"Target threshold: {threshold}%",
            )
            table.add_row(
                "Relevance Score",
                render_progress_bar(rel_score),
                "Topical alignment with research query",
            )
            table.add_row(
                "Factual Grounding",
                render_progress_bar(ground_score),
                "Source density & hallucination resistance",
            )
            table.add_row(
                "Decision Gate",
                "[bold green]✅ APPROVED (Passes Threshold)[/bold green]"
                if (score >= threshold or is_suff)
                else "[bold red]❌ NEEDS REFINEMENT (Looping)[/bold red]",
                "Meets publication criteria" if is_suff else "Triggers follow-up re-planning",
            )

            console.print(table)
            console.print(f"    [bold yellow]Auditor Feedback:[/bold yellow] {feedback}")
            if gaps:
                console.print(f"    [dim red]Identified Knowledge Gaps: {', '.join(gaps)}[/dim red]")

        elif etype == GraphEventType.REPLANNING:
            console.print(
                f"    [bold orange3]⚡ [ROUTER / RE-PLAN][/bold orange3] {event.message}"
            )

        elif etype == GraphEventType.SYNTHESIS_COMPLETED:
            title = data.get("title", "Research Synthesis")
            score = data.get("final_confidence_score", 0)
            console.print(
                f"    [bold magenta]🎉 Report Compiled:[/bold magenta] [bold]'{title}'[/bold] (Confidence: [bold green]{score}%[/bold green])"
            )

        elif etype == GraphEventType.GRAPH_FAILED:
            console.print(f"[bold red]❌ DAG Execution Error:[/bold red] {event.message}")

    return handle_event


def parse_arguments(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI command line arguments."""
    parser = argparse.ArgumentParser(
        description="Local Multi-Agent AI Research Assistant (CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="The research query or topic to investigate. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:3b",
        help="Ollama model name to use for agent reasoning.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force MockLLMClient for offline execution without active GPU/Ollama.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=75,
        help="Critic confidence score threshold (0-100) to authorize report synthesis.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3,
        help="Maximum feedback iterations before enforcing final synthesis.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="File path to save the final generated Markdown report.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="File path to save the full JSON execution state and report.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )

    return parser.parse_args(args)


def run_cli(args: argparse.Namespace) -> int:
    """Core CLI execution handler."""
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Header Banner
    header_text = Text()
    header_text.append("🚀 Local Multi-Agent AI Research Assistant\n", style="bold cyan")
    header_text.append("Autonomous Graph-Driven Research Loop | 3B LLMs + Ollama + Critic Gate", style="dim white")
    console.print(Panel(header_text, border_style="cyan", padding=(1, 2)))

    # Acquire Query
    query = args.query
    if not query or not query.strip():
        try:
            query = Prompt.ask("[bold cyan]Enter your research topic/query[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session aborted by user.[/yellow]")
            return 1

    if not query or not query.strip():
        console.print("[red]Error: Query cannot be empty.[/red]")
        return 1

    # Configuration Display
    config_table = Table(show_header=False, box=None)
    config_table.add_column("Key", style="bold")
    config_table.add_column("Value", style="cyan")
    config_table.add_row("🎯 Research Topic", query.strip())
    config_table.add_row("🤖 LLM Engine", args.model)
    config_table.add_row("⚙️ Execution Mode", "[yellow]Offline Mock Simulator[/yellow]" if args.mock else "[green]Live Ollama Engine[/green]")
    config_table.add_row("📊 Quality Threshold", f"{args.threshold}% Confidence")
    config_table.add_row("🔄 Max Feedback Loops", str(args.max_iter))
    console.print(Panel(config_table, title="[bold]Workflow Configuration[/bold]", border_style="blue"))

    # Initialize LLM Client and Graph
    llm_client = get_llm_client(model_name=args.model, force_mock=args.mock)
    graph = ResearchGraph(llm_client=llm_client)

    # Attach Rich real-time callback
    event_listener = create_event_listener(console, max_iterations=args.max_iter, threshold=args.threshold)
    graph.add_callback(event_listener)

    start_time = time.time()
    console.print(Rule("[bold green]🚀 Launching Autonomous DAG Execution Loop[/bold green]", style="green"))

    # Run Graph
    try:
        state: ResearchState = graph.run(
            query=query.strip(),
            max_iterations=args.max_iter,
            confidence_threshold=args.threshold,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Research run interrupted by user.[/yellow]")
        return 1
    except Exception as err:
        console.print(f"\n[bold red]Research execution failed:[/bold red] {err}")
        return 1

    elapsed = time.time() - start_time

    # Display Rendered Final Markdown Report
    if state.final_report_markdown:
        console.print()
        console.print(Rule("[bold magenta]📄 Final Synthesized Research Report[/bold magenta]", style="magenta"))
        console.print(
            Panel(
                Markdown(state.final_report_markdown),
                title=f"[bold green]Verified Research Synthesis (Score: {state.final_synthesis.final_confidence_score if state.final_synthesis else 'N/A'}%)[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Summary Card
    summary_table = Table(show_header=False, box=None)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", style="cyan")
    summary_table.add_row("⏱️ Total Elapsed Time", f"{elapsed:.2f}s")
    summary_table.add_row("🔄 Feedback Iterations", f"{state.iteration + 1} / {state.max_iterations}")
    summary_table.add_row("🌐 Verified Sources", str(len(state.collected_sources)))
    summary_table.add_row("💡 Grounded Facts", str(len(state.extracted_facts)))
    final_score = state.final_synthesis.final_confidence_score if state.final_synthesis else (state.latest_critique.confidence_score if state.latest_critique else 0)
    summary_table.add_row("🏆 Final Confidence", render_progress_bar(final_score))
    console.print(Panel(summary_table, title="[bold green]Execution Summary[/bold green]", border_style="green"))

    # Save Output Files if requested
    if args.output_file and state.final_report_markdown:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(state.final_report_markdown, encoding="utf-8")
        console.print(f"[green]💾 Markdown report saved to:[/green] [bold]{out_path.resolve()}[/bold]")

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_content = state.model_dump_json(indent=2)
        json_path.write_text(json_content, encoding="utf-8")
        console.print(f"[green]💾 Full JSON state trace saved to:[/green] [bold]{json_path.resolve()}[/bold]")

    console.print()
    console.print("[bold green]✨ Research workflow completed successfully![/bold green]")
    console.print("[dim]💡 Tip: Launch the interactive Web Dashboard anytime via: [bold cyan]streamlit run app.py[/bold cyan][/dim]")
    return 0


def main():
    """Main CLI entry point."""
    args = parse_arguments()
    sys.exit(run_cli(args))


if __name__ == "__main__":
    main()
