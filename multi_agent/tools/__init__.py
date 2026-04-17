#!/usr/bin/env python3
"""
Tools for Hermes Agent
"""

from .rag_tool import RAGRetrievalTool, RAGIndexTool
from .dmb_tool import DMBSearchTool, DMBStoreTool, DMBBestTool, DMBStatsTool
from .simulation_tool import (
    GenerateSysMLTool,
    GenerateMATLABTool,
    RunSimulationTool,
    ParameterStudyTool,
    OptimizeParametersTool,
)
from .memory_tool import MemorySearchTool, MemoryStoreTool, MemoryStatsTool
from .optimization_tool import OptimizationTool

__all__ = [
    # RAG Tools
    "RAGRetrievalTool",
    "RAGIndexTool",
    # DMB Tools
    "DMBSearchTool",
    "DMBStoreTool",
    "DMBBestTool",
    "DMBStatsTool",
    # Simulation Tools
    "GenerateSysMLTool",
    "GenerateMATLABTool",
    "RunSimulationTool",
    "ParameterStudyTool",
    "OptimizeParametersTool",
    # Memory Tools
    "MemorySearchTool",
    "MemoryStoreTool",
    "MemoryStatsTool",
    # Optimization Tools
    "OptimizationTool",
]
