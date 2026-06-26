"""L4: LLM 摘要压缩

包装 Hermes 现有的 ContextCompressor，对中间段消息做逻辑链摘要。
复用 ContextCompressor 的 _generate_summary 方法（摘要生成）、
_serialize_for_summary（序列化）、_compute_summary_budget（预算计算）等能力。
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from agent.context_compressor import ContextCompressor

logger = logging.getLogger(__name__)

# 压缩摘要元数据键——与 ContextCompressor 保持一致
COMPRESSED_SUMMARY_METADATA_KEY = "_compressed_summary"

# 摘要消息中附带哪些 msg_id 被压缩的索引键
COMPRESSED_MSG_IDS_KEY = "_compressed_msg_ids"


def compress_middle(
    middle_msgs: List[Dict[str, Any]],
    compressor: ContextCompressor,
    focus_topic: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """对中间段消息做逻辑链摘要。

    包装 ContextCompressor._generate_summary，复用其完整的摘要生成能力：
    - 结构化模板（Goal / Completed Actions / Key Decisions / ...）
    - 增量摘要（_previous_summary 追踪）
    - 辅助模型 / 主模型回退
    - 防抖 + 故障恢复
    - 自动 focus topic 推导

    Args:
        middle_msgs: 中间段消息列表
        compressor: 已初始化的 ContextCompressor 实例
        focus_topic: 可选 focus topic 字符串

    Returns:
        (summary_message, summary_text)
        summary_message 是单条消息 dict，可插入到压缩后的消息列表
        summary_text 是纯文本摘要
    """
    if not middle_msgs:
        logger.info("L4 压缩跳过：中间段为空")
        return [], ""

    # 使用 ContextCompressor 的 _generate_summary 生成摘要
    summary_text = compressor._generate_summary(middle_msgs, focus_topic=focus_topic)

    if not summary_text:
        logger.warning("L4 摘要生成失败，尝试静态回退")
        # 尝试使用内置的静态回退
        try:
            summary_text = compressor._build_static_fallback_summary(
                middle_msgs,
                reason="LLM summary generation returned empty",
            )
        except Exception as e:
            logger.error("L4 静态回退也失败: %s", e)
            return [], ""

    # 构建摘要消息
    summary_msg = _build_summary_message(summary_text)

    logger.info("L4 压缩完成：%d 条消息 → 1 条摘要 (%d chars)",
                len(middle_msgs), len(summary_text))

    return [summary_msg], summary_text


def _build_summary_message(summary_text: str) -> Dict[str, Any]:
    """构建摘要消息，带元数据标记。

    Args:
        summary_text: 摘要文本

    Returns:
        摘要消息 dict
    """
    return {
        "role": "assistant",
        "content": summary_text,
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }


def add_compressed_msg_ids(summary_message: Dict[str, Any], msg_ids: List[str]) -> Dict[str, Any]:
    """在摘要消息上附加被压缩的 msg_id 列表。

    Args:
        summary_message: 摘要消息
        msg_ids: 被压缩的消息 ID 列表

    Returns:
        修改后的摘要消息
    """
    return {**summary_message, COMPRESSED_MSG_IDS_KEY: msg_ids}
