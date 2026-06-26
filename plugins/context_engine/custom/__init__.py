"""Write the hermes_custom __init__.py"""

import json, logging, os
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

from .tool_protector import protect_tool_arguments, restore_tool_arguments, compute_arg_hash
from .segmenter import segment_messages
from .ccr_store import CcrStore
from .compressor import compress_middle, add_compressed_msg_ids, COMPRESSED_MSG_IDS_KEY
from .validator import validate_compression
from .retriever import retrieve_original, get_compressed_index, handle_retrieve_tool

logger = logging.getLogger(__name__)

MAX_COMPRESSION_COUNT = 2


def register(ctx):
    engine = HermesCustomEngine()
    ctx.register_context_engine(engine)


class HermesCustomEngine(ContextEngine):

    @property
    def name(self) -> str:
        return "custom"

    def __init__(self):
        super().__init__()
        self._session_id: str = ""
        self._compressor = None
        self._agent = None
        self._compression_count_session: int = 0
        self._config_threshold_percent: float = 0.42
        self._last_compressed_msg_ids: List[str] = []
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.threshold_tokens = 0
        self.context_length = 0
        self.compression_count = 0
        self._last_compress_aborted: bool = False
        self._ineffective_compression_count: int = 0
        self.protect_first_n = 1
        self.protect_last_n = 15
        self.tool_result_threshold = 1500
        self.skip_short = 250

    def _init_from_agent(self, agent) -> None:
        self._agent = agent
        # Create a real ContextCompressor instance (not self-referencing)
        from agent.context_compressor import ContextCompressor
        config = getattr(agent, "config", None) or {}
        comp_cfg = config.get("compression", {}) if isinstance(config, dict) else {}
        self._compressor = ContextCompressor(
            model=comp_cfg.get("model", agent.model),
            threshold_percent=self._config_threshold_percent,
            base_url=getattr(agent, "base_url", ""),
            api_key=getattr(agent, "api_key", ""),
            provider=getattr(agent, "provider", ""),
            api_mode=getattr(agent, "api_mode", ""),
        )
        self.context_length = self._compressor.context_length
        self.threshold_tokens = self._compressor.threshold_tokens
        logger.info(
            "HermesCustomEngine: created ContextCompressor (model=%s, ctx=%d, thresh=%d, thr%%=%.0f%%)",
            getattr(self._compressor, "model", "?"),
            self.context_length, self.threshold_tokens,
            self._config_threshold_percent * 100,
        )
        config = getattr(agent, "config", None) or {}
        context_cfg = config.get("context", {}) if isinstance(config, dict) else {}
        if isinstance(context_cfg, dict):
            engine_cfg = context_cfg.get("engine_config", {})
            if isinstance(engine_cfg, dict):
                self.protect_first_n = int(engine_cfg.get("protect_first_n", self.protect_first_n))
                self.protect_last_n = int(engine_cfg.get("protect_last_n", self.protect_last_n))
                self.tool_result_threshold = int(engine_cfg.get("tool_result_threshold", self.tool_result_threshold))

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        """Override to use _config_threshold_percent (0.42) instead of the default 0.75."""
        self.threshold_percent = self._config_threshold_percent
        super().update_model(model, context_length, base_url=base_url,
                             api_key=api_key, provider=provider, api_mode=api_mode)

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", self.last_prompt_tokens + self.last_completion_tokens)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if tokens < self.threshold_tokens:
            return False
        if self._compression_count_session >= MAX_COMPRESSION_COUNT:
            logger.info("Max session compression (%d) reached, skip", MAX_COMPRESSION_COUNT)
            return False
        # Anti-thrashing: back off if recent compressions were ineffective
        if self._ineffective_compression_count >= 2:
            logger.warning(
                "Compression skipped — last %d compressions failed or saved <10%%. "
                "Consider /new to start a fresh session, or /compress <topic> "
                "for focused compression.",
                self._ineffective_compression_count,
            )
            return False
        return True

    def compress(self, messages, current_tokens=None, focus_topic=None):
        if not messages:
            return messages
        # Reset per-call abort flag so compress_context can detect failures
        self._last_compress_aborted = False
        logger.info("=" * 60)
        logger.info("CustomEngine compress start (%d msgs)", len(messages))
        if self._compressor is None and self._agent is not None:
            self._init_from_agent(self._agent)
        def _warn(msg):
            logger.warning(msg)
            if self._agent and hasattr(self._agent, "_emit_warning"):
                try:
                    self._agent._emit_warning(f"⚠ {msg}")
                except Exception:
                    pass

        # L1
        logger.info("[L1] Protect tool arguments...")
        protected_msgs, protected_map = protect_tool_arguments(messages)
        original_hashes = compute_arg_hash(messages)

        # L2
        logger.info("[L2] Segment messages...")
        head_msgs, middle_msgs, tail_msgs = segment_messages(
            protected_msgs,
            protect_first_n=self.protect_first_n,
            protect_last_n=self.protect_last_n,
            tool_result_threshold=self.tool_result_threshold,
            skip_short=self.skip_short,
        )
        if not middle_msgs:
            logger.info("No middle segment, return original")
            return messages

        # L3
        logger.info("[L3] CCR store original messages...")
        session_id = self._session_id or "default"
        msg_ids = CcrStore.store_batch(session_id, middle_msgs)
        self._last_compressed_msg_ids = msg_ids

        # L4
        logger.info("[L4] LLM summarization...")
        if self._compressor is not None:
            summary_msgs, summary_text = compress_middle(
                middle_msgs, compressor=self._compressor, focus_topic=focus_topic,
            )
        else:
            _warn("压缩引擎未初始化，跳过摘要")
            summary_msgs = []
        if not summary_msgs:
            _warn("摘要生成失败，已自动回退。对话不受影响，但 token 会继续增长。建议 /new 或手动 /compress 重试。")
            return messages
        summary_msgs[0] = add_compressed_msg_ids(summary_msgs[0], msg_ids)
        CcrStore.store_compression_record(session_id, msg_ids)

        # 角色协商：避免连续同 role（移植自 ContextCompressor.compress()）
        last_head_role = head_msgs[-1].get("role", "user") if head_msgs else "user"
        first_tail_role = tail_msgs[0].get("role", "user") if tail_msgs else "user"

        summary_role = "user" if last_head_role in {"assistant", "tool"} else "assistant"

        if summary_role == first_tail_role:
            flipped = "user" if summary_role == "assistant" else "assistant"
            if flipped != last_head_role:
                summary_role = flipped

        # 应用角色到摘要消息
        if summary_msgs:
            summary_msgs[0]["role"] = summary_role

        compressed_result = list(head_msgs) + summary_msgs + list(tail_msgs)

        # L5
        logger.info("[L5] Restore + validate...")
        restored_msgs = restore_tool_arguments(compressed_result, protected_map)
        if not validate_compression(messages, restored_msgs, protected_map, original_hashes):
            _warn("压缩验证失败，已自动回退。对话不受影响，但 token 会继续增长。建议 /new 或手动 /compress 重试。")
            self._last_compress_aborted = True
            self._ineffective_compression_count += 1
            self._compression_count_session -= 1
            return messages

        self.compression_count += 1
        self._compression_count_session += 1
        self._ineffective_compression_count = 0  # success resets counter
        logger.info("Compress done: %d -> %d (saved %d)",
                     len(messages), len(restored_msgs), len(messages) - len(restored_msgs))
        logger.info("=" * 60)
        return restored_msgs

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._compression_count_session = 0
        self._ineffective_compression_count = 0
        self._last_compressed_msg_ids = []
        agent = kwargs.get("agent")
        if agent is not None:
            self._init_from_agent(agent)
        CcrStore.cleanup_old()
        logger.info("Session start: %s", session_id)

    def on_session_end(self, session_id: str, messages=None) -> None:
        logger.info("Session end: %s", session_id)

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._compression_count_session = 0
        self._ineffective_compression_count = 0
        self._last_compressed_msg_ids = []
        if self._session_id:
            CcrStore.clear_session(self._session_id)

    def has_content_to_compress(self, messages) -> bool:
        return len(messages) > self.protect_first_n + self.protect_last_n + 1

    def get_tool_schemas(self):
        return [
            {
                "name": "retrieve_original_content",
                "description": "Retrieve original message from compression CCR store",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID"},
                        "msg_id": {"type": "string", "description": "Message ID to retrieve"},
                    },
                    "required": ["session_id", "msg_id"],
                },
            },
            {
                "name": "retrieve_compressed_index",
                "description": "List compressed message IDs for a session",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID"},
                    },
                    "required": ["session_id"],
                },
            },
            {
                "name": "ccr_search",
                "description": "Search compressed original messages by keyword. Only works on messages that were compacted in the CURRENT session. For old session history use session_search instead.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "Keyword to search for in compacted messages"},
                        "session_id": {"type": "string", "description": "Optional: limit search to a specific session ID"},
                        "limit": {"type": "integer", "description": "Max results to return (default 10)", "default": 10},
                    },
                    "required": ["keyword"],
                },
            },
        ]

    def handle_tool_call(self, name, args, **kwargs):
        if name == "retrieve_original_content":
            sid = args.get("session_id", self._session_id or "default")
            return handle_retrieve_tool(sid, args.get("msg_id", ""))
        if name == "retrieve_compressed_index":
            sid = args.get("session_id", self._session_id or "default")
            return json.dumps(get_compressed_index(sid), ensure_ascii=False)
        if name == "ccr_search":
            from .ccr_search import ccr_search as search_fn
            return search_fn(
                keyword=args.get("keyword", ""),
                session_id=args.get("session_id", None),
                limit=args.get("limit", 10),
            )
        return json.dumps({"error": f"Unknown tool: {name}"})

    def get_status(self):
        s = super().get_status()
        s["engine"] = self.name
        s["compression_count_session"] = self._compression_count_session
        s["max_compression_per_session"] = MAX_COMPRESSION_COUNT
        s["compressed_msg_ids_count"] = len(self._last_compressed_msg_ids)
        return s
