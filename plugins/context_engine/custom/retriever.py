"""L6: 检索能力

从 CCR 存储中按需找回原始消息。
提供 commands 和函数两种使用方式。
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .ccr_store import CcrStore

logger = logging.getLogger(__name__)


def retrieve_original(session_id: str, msg_id: str) -> Optional[Dict[str, Any]]:
    """按需检索原始消息。

    Args:
        session_id: 会话 ID
        msg_id: 消息 ID

    Returns:
        原始消息 dict，未找到时返回 None
    """
    return CcrStore.retrieve_original(session_id, msg_id)


def retrieve_batch(
    session_id: str,
    msg_ids: List[str],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """批量检索原始消息。

    Args:
        session_id: 会话 ID
        msg_ids: 消息 ID 列表

    Returns:
        {msg_id: original_message_or_None}
    """
    return CcrStore.retrieve_batch(session_id, msg_ids)


def get_compressed_index(session_id: str) -> Dict[str, Any]:
    """获取指定 session 的压缩索引。

    返回哪些 msg_id 被压缩了，以及对应的摘要信息。

    Args:
        session_id: 会话 ID

    Returns:
        {
            "session_id": session_id,
            "compressed_msg_ids": [...],
            "count": int,
        }
    """
    compressed_ids = CcrStore.list_compressed_ids(session_id)
    return {
        "session_id": session_id,
        "compressed_msg_ids": compressed_ids,
        "count": len(compressed_ids),
    }


def handle_retrieve_tool(
    session_id: str,
    msg_id: str,
    format_output: bool = True,
) -> str:
    """将 retrieve_original 包装为 tool call 处理函数。

    Args:
        session_id: 会话 ID
        msg_id: 消息 ID
        format_output: 是否格式化输出（True=JSON 字符串，False=dict）

    Returns:
        JSON 字符串结果
    """
    result = retrieve_original(session_id, msg_id)
    if result is None:
        return json.dumps({
            "found": False,
            "session_id": session_id,
            "msg_id": msg_id,
            "error": "未找到原始消息，可能已过期或从未压缩",
        }, ensure_ascii=False)

    return json.dumps({
        "found": True,
        "session_id": session_id,
        "msg_id": msg_id,
        "message": result,
    }, ensure_ascii=False)
