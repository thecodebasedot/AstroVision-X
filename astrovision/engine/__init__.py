"""The analysis engine: pipeline orchestration, ranking and narrative."""

from .assistant import DISCOVERY_DISCLAIMER, ResearchAssistant
from .pipeline import Pipeline, StageResult
from .priority import PRIORITY_WEIGHTS, PriorityItem, rank_candidates

__all__ = [
    "Pipeline", "StageResult",
    "ResearchAssistant", "DISCOVERY_DISCLAIMER",
    "rank_candidates", "PriorityItem", "PRIORITY_WEIGHTS",
]
