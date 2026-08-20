"""
Graph Engine and DAG Execution State Machine module for Local Multi-Agent Research Assistant.
Exports ResearchGraph, GraphEvent, and GraphEventType.
"""

from local_researcher.graph.dag import (
    ResearchGraph,
    GraphEvent,
    GraphEventType,
    format_report_to_markdown,
)

__all__ = [
    "ResearchGraph",
    "GraphEvent",
    "GraphEventType",
    "format_report_to_markdown",
]
