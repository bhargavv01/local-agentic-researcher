"""
Event contracts and schemas for ResearchGraph execution lifecycle and streaming callbacks.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class GraphEventType(str, Enum):
    """Types of streaming events emitted during DAG graph execution."""
    GRAPH_STARTED = "graph_started"
    GRAPH_COMPLETED = "graph_completed"
    GRAPH_FAILED = "graph_failed"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    STEP_UPDATE = "step_update"
    ROUTER_DECISION = "router_decision"
    ITERATION_STARTED = "iteration_started"
    CRITIQUE_EVALUATED = "critique_evaluated"
    REPLANNING = "replanning"
    SYNTHESIS_COMPLETED = "synthesis_completed"


class GraphEvent(BaseModel):
    """Structured event payload emitted during DAG state machine transitions."""
    node_name: str = Field(default="graph", description="Name of the node or component emitting the event")
    event_type: GraphEventType | str = Field(..., description="Classification of the event")
    message: str = Field(default="", description="Human-readable event description")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data and state details",
    )

    def __init__(
        self,
        node_name: str = "graph",
        event_type: GraphEventType | str = GraphEventType.STEP_UPDATE,
        message: str = "",
        timestamp: str | None = None,
        payload: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        iteration: int | None = None,
        **kwargs: Any,
    ):
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        if payload is None:
            payload = data or {}
        if iteration is not None and "iteration" not in payload:
            payload["iteration"] = iteration
        super().__init__(
            node_name=node_name,
            event_type=event_type,
            message=message,
            timestamp=timestamp,
            payload=payload,
            **kwargs,
        )

    @property
    def data(self) -> dict[str, Any]:
        """Alias for payload."""
        return self.payload

    @property
    def iteration(self) -> int:
        """Iteration count."""
        return self.payload.get("iteration", 0)
