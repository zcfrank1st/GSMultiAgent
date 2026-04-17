#!/usr/bin/env python3
"""
DMB (Dynamic Memory Buffer) Tools for Hermes Agent
Provides experience storage and retrieval capabilities
"""

import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class DMBToolMixin:
    """Mixin class to add DMB capabilities to tools"""

    def __init__(self, *args, **kwargs):
        self.dmb = None
        super().__init__(*args, **kwargs)

    def set_dmb(self, dmb) -> None:
        """Set the DMB memory buffer"""
        self.dmb = dmb


class DMBSearchTool(DMBToolMixin):
    """Tool for retrieving similar experiences from DMB"""

    name = "dmb_search"
    description = """
    Search for similar experiences in the dynamic memory buffer.
    Use this when you need to find historical cases with similar task context.
    Returns best matching experiences with fitness scores.
    """

    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "object", "description": "Task context to search for"},
            "top_k": {
                "type": "integer",
                "description": "Number of similar experiences to retrieve",
                "default": 5,
            },
            "memory_type": {
                "type": "string",
                "description": "Type of memory: short_term, long_term, or all",
                "enum": ["short_term", "long_term", "all"],
                "default": "all",
            },
        },
        "required": ["query"],
    }

    async def execute(
        self, query: Dict[str, Any], top_k: int = 5, memory_type: str = "all"
    ) -> Dict[str, Any]:
        """Execute DMB search"""
        if not self.dmb:
            return {"status": "error", "message": "DMB not initialized"}

        try:
            from ..memory.dmb import MemoryType

            mem_type = None
            if memory_type == "short_term":
                mem_type = MemoryType.SHORT_TERM
            elif memory_type == "long_term":
                mem_type = MemoryType.LONG_TERM

            results = await self.dmb.retrieve_similar(
                query=query, top_k=top_k, memory_type=mem_type
            )

            return {
                "status": "success",
                "query": query,
                "results_count": len(results),
                "experiences": results,
            }
        except Exception as e:
            logger.error(f"DMB search failed: {e}")
            return {"status": "error", "message": str(e)}


class DMBStoreTool(DMBToolMixin):
    """Tool for storing experiences in DMB"""

    name = "dmb_store"
    description = """
    Store a new experience in the dynamic memory buffer.
    Use this after successful optimization to save best parameters and results.
    """

    input_schema = {
        "type": "object",
        "properties": {
            "task_context": {"type": "object", "description": "Task context information"},
            "parameters": {"type": "object", "description": "Optimization parameters"},
            "objectives": {"type": "object", "description": "Objective values achieved"},
            "fitness": {"type": "number", "description": "Overall fitness score"},
            "memory_type": {
                "type": "string",
                "description": "short_term or long_term memory",
                "enum": ["short_term", "long_term"],
                "default": "short_term",
            },
        },
        "required": ["task_context", "parameters", "fitness"],
    }

    async def execute(
        self,
        task_context: Dict[str, Any],
        parameters: Dict[str, float],
        objectives: Dict[str, float],
        fitness: float,
        memory_type: str = "short_term",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute DMB store"""
        if not self.dmb:
            return {"status": "error", "message": "DMB not initialized"}

        try:
            from ..memory.dmb import MemoryType

            mem_type = MemoryType.LONG_TERM if memory_type == "long_term" else MemoryType.SHORT_TERM

            memory_id = await self.dmb.store(
                task_context=task_context,
                parameters=parameters,
                objectives=objectives,
                fitness=fitness,
                memory_type=mem_type,
                metadata=metadata,
            )

            return {"status": "success", "memory_id": memory_id}
        except Exception as e:
            logger.error(f"DMB store failed: {e}")
            return {"status": "error", "message": str(e)}


class DMBBestTool(DMBToolMixin):
    """Tool for retrieving best performing experiences"""

    name = "dmb_best"
    description = """
    Retrieve the best performing experiences for a given task context.
    Use this to leverage proven best solutions for similar tasks.
    """

    input_schema = {
        "type": "object",
        "properties": {
            "task_context": {"type": "object", "description": "Task context to match against"},
            "top_k": {
                "type": "integer",
                "description": "Number of best experiences to retrieve",
                "default": 5,
            },
        },
        "required": ["task_context"],
    }

    async def execute(self, task_context: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """Execute DMB best retrieval"""
        if not self.dmb:
            return {"status": "error", "message": "DMB not initialized"}

        try:
            results = await self.dmb.retrieve_best(task_context=task_context, top_k=top_k)

            return {
                "status": "success",
                "task_context": task_context,
                "results_count": len(results),
                "best_experiences": results,
            }
        except Exception as e:
            logger.error(f"DMB best retrieval failed: {e}")
            return {"status": "error", "message": str(e)}


class DMBStatsTool(DMBToolMixin):
    """Tool for getting DMB statistics"""

    name = "dmb_stats"
    description = """
    Get statistics about the dynamic memory buffer.
    Use this to understand memory usage and performance.
    """

    input_schema = {"type": "object", "properties": {}}

    async def execute(self) -> Dict[str, Any]:
        """Execute DMB stats"""
        if not self.dmb:
            return {"status": "error", "message": "DMB not initialized"}

        try:
            stats = await self.dmb.get_statistics()

            return {"status": "success", "statistics": stats}
        except Exception as e:
            logger.error(f"DMB stats failed: {e}")
            return {"status": "error", "message": str(e)}
