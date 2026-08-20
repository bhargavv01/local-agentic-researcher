"""
Interactive Rich Command-Line Interface for Local Multi-Agent Research Assistant.
Provides real-time visualization of DAG node execution, critic score gauges,
interactive prompting, and Markdown / JSON report export.
"""

import argparse
import logging
import sys
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


def create_event_listener(console: Console, max_iterations: int, threshold: int):
    """
    Creates an event callback handler to render DAG node events in real-time.
    """

    def handle_event(event: GraphEvent):
        etype = event.event_type
        data = event.data

        if etype == GraphEventType.ITERATION_STARTED:
            console.print()
            console.print(
                Rule(
                    f"[bold cyan]DAG Feedback Loop — Iteration {event.iteration} of {max_iterations}[/bold cyan]",
                    style="cyan",
                )
            )

        elif etype == GraphEventType.NODE_STARTED:
            node_labels = {
                "planner": "[bold blue][PLANNER][/bold blue]",
                "researcher": "[bold green][RESEARCHER][/bold green]",
                "critic": "[bold yellow][CRITIC][/bold yellow]",
                "synthesizer": "[bold magenta][SYNTHESIZER][/bold magenta]",
            }
            label = node_labels.get(event.node_name, f"[bold][{event.node_name.upper()}][/bold]")
            console.print(f"  {label} {event.message}...")

        elif etype == GraphEventType.NODE_COMPLETED:
            if event.node_name == "planner" and "search_queries" in data:
                queries = data.get("search_queries", [])
                table = Table(title="Generated Search Queries", show_header=True, header_style="bold blue")
                table.add_column("#", style="dim", width=4)
                table.add_column("Query", style="cyan")
                for idx, q in enumerate(queries, start=1):
                    table.add_row(str(idx), q)
                console.print(table)

            elif event.node_name == "researcher" and "total_sources" in data:
                total_sources = data.get("total_sources", 0)
                total_facts = data.get("total_facts", 0)
                console.print(
                    f"    [green][OK] Research completed:[/green] [bold]{total_sources}[/bold] sources aggregated, "
                    f"[bold]{total_facts}[/bold] grounded facts extracted."
                )

        elif etype == GraphEventType.CRITIQUE_EVALUATED:
            score = data.get("confidence_score", 0)
            rel_score = data.get("relevance_score", 0)
            ground_score = data.get("factual_grounding_score", 0)
            is_suff = data.get("is_sufficient", False)
            feedback = data.get("feedback", "")
            gaps = data.get("identified_gaps", [])

            score_color = "green" if score >= threshold else "yellow" if score >= 50 else "red"

            table = Table(title="Critic Quality Assessment", show_header=True, header_style="bold yellow")
            table.add_column("Metric", style="bold")
            table.add_column("Score / Status", justify="center")
            table.add_column("Details", style="italic")

            table.add_row(
                "Overall Confidence",
                f"[{score_color}][bold]{score}/100[/bold][/{score_color}]",
                f"Threshold: {threshold}",
            )
            table.add_row("Relevance Score", f"{rel_score}/100", "Topical alignment with user query")
            table.add_row("Factual Grounding", f"{ground_score}/100", "Verification and source density")
            table.add_row(
                "Quality Gate Passed",
                "[green]YES[/green]" if (score >= threshold or is_suff) else "[red]NO[/red]",
                "Meets publication standards" if is_suff else "Requires refinement",
            )

            console.print(table)
            console.print(f"    [yellow]Feedback:[/yellow] {feedback}")
            if gaps:
                console.print(f"    [dim]Identified Gaps: {', '.join(gaps)}[/dim]")

        elif etype == GraphEventType.REPLANNING:
            console.print(
                f"    [bold orange3][RE-PLAN] Feedback loop triggered:[/bold orange3] {event.message}"
            )

        elif etype == GraphEventType.SYNTHESIS_COMPLETED:
            title = data.get("title", "Research Synthesis")
            score = data.get("final_confidence_score", 0)
            console.print(
                f"    [magenta][OK] Report Compiled:[/magenta] [bold]'{title}'[/bold] (Final Confidence: [bold]{score}%[/bold])"
            )

        elif etype == GraphEventType.GRAPH_FAILED:
            console.print(f"[bold red][ERROR] DAG Execution Error:[/bold red] {event.message}")

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
    header_text.append("Local Multi-Agent Research Assistant\n", style="bold cyan")
    header_text.append("Autonomous Graph-Driven Research Loop | Ollama & Small Models", style="dim")
    console.print(Panel(header_text, border_style="cyan"))

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
    config_table.add_row("Research Query", query.strip())
    config_table.add_row("Model", args.model)
    config_table.add_row("Mode", "[yellow]Offline Mock Mode[/yellow]" if args.mock else "[green]Live Ollama Engine[/green]")
    config_table.add_row("Confidence Threshold", f"{args.threshold}/100")
    config_table.add_row("Max Feedback Loops", str(args.max_iter))
    console.print(Panel(config_table, title="[bold]Configuration[/bold]", border_style="blue"))

    # Initialize LLM Client and Graph
    llm_client = get_llm_client(model_name=args.model, force_mock=args.mock)
    graph = ResearchGraph(llm_client=llm_client)

    # Attach Rich real-time callback
    event_listener = create_event_listener(console, max_iterations=args.max_iter, threshold=args.threshold)
    graph.add_callback(event_listener)

    console.print(Rule("[bold green]Executing Multi-Agent DAG Loop[/bold green]", style="green"))

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

    # Display Rendered Final Markdown Report
    if state.final_report_markdown:
        console.print()
        console.print(Rule("[bold magenta]Final Synthesized Research Report[/bold magenta]", style="magenta"))
        console.print(
            Panel(
                Markdown(state.final_report_markdown),
                title="[bold green]Report Preview[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Save Output Files if requested
    if args.output_file and state.final_report_markdown:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(state.final_report_markdown, encoding="utf-8")
        console.print(f"[green][OK] Markdown report saved to:[/green] [bold]{out_path.resolve()}[/bold]")

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_content = state.model_dump_json(indent=2)
        json_path.write_text(json_content, encoding="utf-8")
        console.print(f"[green][OK] Full JSON state saved to:[/green] [bold]{json_path.resolve()}[/bold]")

    console.print()
    console.print("[bold green][DONE] Research workflow completed successfully![/bold green]")
    return 0


def main():
    """Main CLI entry point."""
    args = parse_arguments()
    sys.exit(run_cli(args))


if __name__ == "__main__":
    main()
