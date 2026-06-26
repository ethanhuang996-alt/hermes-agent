"""L2: 分段划定

将消息列表分为三段：
- head_msgs: 前 N 条，全文保留
- middle_msgs: 中间段，进入压缩管道
- tail_msgs: 后 M 条，全文保留（最新对话）
"""

import logging
from typing import Any, Dict, List, Tuple

from agent.model_metadata import estimate_messages_tokens_rough

logger = logging.getLogger(__name__)


def segment_messages(
    messages: List[Dict[str, Any]],
    protect_first_n: int = 1,
    protect_last_n: int = 15,
    tool_result_threshold: int = 1500,
    skip_short: int = 250,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """分段划定消息列表。

    Args:
        messages: 完整消息列表（不含 system prompt，已由 Hermes 独立处理）
        protect_first_n: 头部保留条数
        protect_last_n: 尾部保留条数
        tool_result_threshold: 超过此 token 数的 tool_result 才压缩
        skip_short: 低于此 token 数的消息跳过压缩

    Returns:
        (head_msgs, middle_msgs, tail_msgs)
    """
    n = len(messages)
    if n == 0:
        return [], [], []

    # 头部保留
    head_count = min(protect_first_n, n)
    head_msgs = messages[:head_count]

    # 尾部保留
    tail_count = min(protect_last_n, n - head_count)
    tail_msgs = messages[n - tail_count:] if tail_count > 0 else []

    # 中间段
    middle_end = n - tail_count
    middle_msgs = messages[head_count:middle_end] if head_count < middle_end else []

    if not middle_msgs:
        logger.info("L2 分段划定：无中间段可压缩 (head=%d, tail=%d, total=%d)",
                     head_count, tail_count, n)
        return head_msgs, [], tail_msgs

    # 对中间段做二级筛选
    filtered_middle: List[Dict[str, Any]] = []
    for msg in middle_msgs:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        # tool 结果：超过阈值才压缩
        if role == "tool":
            rough_tokens = estimate_messages_tokens_rough([msg])
            if rough_tokens < tool_result_threshold:
                logger.debug("L2 跳过短 tool_result (%d tokens < %d)", rough_tokens, tool_result_threshold)
                continue  # 保留全文，不进入压缩

        # 短消息跳过
        if isinstance(content, str) and len(content) < skip_short * 2:
            # 粗略字符估计：~4 chars/token
            continue

        filtered_middle.append(msg)

    # 如果过滤后中间段为空，也跳过
    if not filtered_middle:
        logger.info("L2 分段划定：二级筛选后中间段为空")
        return head_msgs, [], tail_msgs

    logger.info(
        "L2 分段划定：head=%d, middle=%d (filtered from %d), tail=%d, total=%d",
        len(head_msgs), len(filtered_middle), len(middle_msgs),
        len(tail_msgs), n,
    )

    return head_msgs, filtered_middle, tail_msgs
