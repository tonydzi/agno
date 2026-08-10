from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from agno.compression._history_compaction import CompactionResult, ContextCompactionManager
from agno.compression._tool_compression import CompressionManager as ToolCompressionManager
from agno.models.base import Model
from agno.models.message import Message
from agno.utils.log import log_debug

if TYPE_CHECKING:
    from agno.metrics import RunMetrics
    from agno.run.agent import RunOutput


@dataclass
class CompressionManager:
    """Unified compression manager for tool results and message history.

    Single entry point for all compression. Internally creates and manages:
    - Tool compression (lightweight, compresses individual tool results)
    - Message compression (heavier, summarizes old conversation history)

    Usage:
        compression_manager = CompressionManager(
            model=OpenAI(id="gpt-4o-mini"),
            compress_tool_results=True,
            compress_messages=True,
        )
    """

    model: Optional[Model] = None

    # Tool compression
    compress_tool_results: bool = True
    compress_tool_results_limit: Optional[int] = None
    compress_token_limit: Optional[int] = None
    compress_tool_call_instructions: Optional[str] = None

    # Message compression
    compress_messages: bool = True
    compress_messages_limit: Optional[int] = None
    compress_messages_token_limit: Optional[int] = None
    compress_messages_keep_recent_count: int = 10
    compress_messages_preserve_user_budget: Optional[int] = None
    compress_messages_instructions: Optional[str] = None

    stats: Dict[str, Any] = field(default_factory=dict)

    # Internal managers
    _tool_compressor: Optional[ToolCompressionManager] = field(default=None, repr=False)
    _history_compactor: Optional[ContextCompactionManager] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.compress_tool_results:
            self._tool_compressor = ToolCompressionManager(
                model=self.model,
                compress_tool_results=True,
                compress_tool_results_limit=self.compress_tool_results_limit,
                compress_token_limit=self.compress_token_limit,
                compress_tool_call_instructions=self.compress_tool_call_instructions,
            )

        if self.compress_messages:
            self._history_compactor = ContextCompactionManager(
                model=self.model,
                message_limit=self.compress_messages_limit,
                token_limit=self.compress_messages_token_limit,
                keep_recent=self.compress_messages_keep_recent_count,
                preserve_user_budget=self.compress_messages_preserve_user_budget,
                instructions=self.compress_messages_instructions,
            )

    # --- Backward compatibility properties ---

    @property
    def compression_manager(self) -> Optional[ToolCompressionManager]:
        """Access internal tool compressor (for _run.py call sites)."""
        return self._tool_compressor

    @property
    def context_compaction_manager(self) -> Optional[ContextCompactionManager]:
        """Access internal history compactor (for _run.py call sites)."""
        return self._history_compactor

    # --- Tool compression (mid-loop) ---

    def should_compress_tools(
        self,
        messages: List[Message],
        tools: Optional[List] = None,
        model: Optional[Model] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> bool:
        """Check if tool results should be compressed."""
        if self._tool_compressor is None:
            return False
        return self._tool_compressor.should_compress(messages, tools, model, response_format)

    async def ashould_compress_tools(
        self,
        messages: List[Message],
        tools: Optional[List] = None,
        model: Optional[Model] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> bool:
        """Async check if tool results should be compressed."""
        if self._tool_compressor is None:
            return False
        return await self._tool_compressor.ashould_compress(messages, tools, model, response_format)

    # Backward compat aliases (base.py calls these)
    should_compress = should_compress_tools
    ashould_compress = ashould_compress_tools

    def compress_tools(
        self,
        messages: List[Message],
        run_metrics: Optional["RunMetrics"] = None,
    ) -> None:
        """Compress tool results in-place (sets msg.compressed_content)."""
        if self._tool_compressor is None:
            return
        log_debug("[COMPRESSION_MANAGER] compress_tools()")
        self._tool_compressor.compress(messages, run_metrics)
        self._sync_stats()

    async def acompress_tools(
        self,
        messages: List[Message],
        run_metrics: Optional["RunMetrics"] = None,
    ) -> None:
        """Async compress tool results in-place."""
        if self._tool_compressor is None:
            return
        log_debug("[COMPRESSION_MANAGER] acompress_tools()")
        await self._tool_compressor.acompress(messages, run_metrics)
        self._sync_stats()

    # --- Message compression (pre-loop) ---

    def should_compress_messages(self, messages: List[Message]) -> bool:
        """Check if messages should be compressed."""
        if self._history_compactor is None:
            return False
        return self._history_compactor.should_compact(messages)

    async def ashould_compress_messages(self, messages: List[Message]) -> bool:
        """Async check if messages should be compressed."""
        if self._history_compactor is None:
            return False
        return await self._history_compactor.ashould_compact(messages)

    def compress_message_history(
        self,
        messages: List[Message],
        run_response: Optional["RunOutput"] = None,
        run_metrics: Optional["RunMetrics"] = None,
    ) -> CompactionResult:
        """Compress message history into summary message."""
        if self._history_compactor is None:
            return CompactionResult(compacted_messages=messages, summary=None)
        log_debug("[COMPRESSION_MANAGER] compress_message_history()")
        result = self._history_compactor.compact(messages, run_response, run_metrics)
        self._sync_stats()
        return result

    async def acompress_message_history(
        self,
        messages: List[Message],
        run_response: Optional["RunOutput"] = None,
        run_metrics: Optional["RunMetrics"] = None,
    ) -> CompactionResult:
        """Async compress message history into summary message."""
        if self._history_compactor is None:
            return CompactionResult(compacted_messages=messages, summary=None)
        log_debug("[COMPRESSION_MANAGER] acompress_message_history()")
        result = await self._history_compactor.acompact(messages, run_response, run_metrics)
        self._sync_stats()
        return result

    # --- Unified compress (cheapest-first escalation) ---

    def compress(
        self,
        messages: List[Message],
        run_response: Optional["RunOutput"] = None,
        run_metrics: Optional["RunMetrics"] = None,
        tools: Optional[List] = None,
        model: Optional[Model] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> CompactionResult:
        """Compress context using cheapest-first escalation.

        1. Tool compression first (lightweight)
        2. Message compression if still needed (heavier)
        """
        # 1. Tool compression
        if self.should_compress_tools(messages, tools, model, response_format):
            self.compress_tools(messages, run_metrics)

        # 2. Message compression
        if self.should_compress_messages(messages):
            return self.compress_message_history(messages, run_response, run_metrics)

        return CompactionResult(compacted_messages=messages, summary=None)

    async def acompress(
        self,
        messages: List[Message],
        run_response: Optional["RunOutput"] = None,
        run_metrics: Optional["RunMetrics"] = None,
        tools: Optional[List] = None,
        model: Optional[Model] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> CompactionResult:
        """Async compress context using cheapest-first escalation."""
        # 1. Tool compression
        if await self.ashould_compress_tools(messages, tools, model, response_format):
            await self.acompress_tools(messages, run_metrics)

        # 2. Message compression
        if await self.ashould_compress_messages(messages):
            return await self.acompress_message_history(messages, run_response, run_metrics)

        return CompactionResult(compacted_messages=messages, summary=None)

    def _sync_stats(self) -> None:
        """Aggregate stats from internal managers."""
        if self._tool_compressor is not None:
            for key, value in self._tool_compressor.stats.items():
                self.stats[f"tool_{key}"] = value
        if self._history_compactor is not None:
            for key, value in self._history_compactor.stats.items():
                self.stats[f"history_{key}"] = value
