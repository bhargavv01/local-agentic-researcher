"""
Specialized research agent modules for the Multi-Agent AI Research Assistant.
"""

from local_researcher.agents.base import BaseAgent
from local_researcher.agents.critic import CriticAgent
from local_researcher.agents.planner import PlannerAgent
from local_researcher.agents.researcher import ResearchAgent
from local_researcher.agents.synthesizer import SynthesizerAgent

__all__ = [
    "BaseAgent",
    "PlannerAgent",
    "ResearchAgent",
    "CriticAgent",
    "SynthesizerAgent",
]
