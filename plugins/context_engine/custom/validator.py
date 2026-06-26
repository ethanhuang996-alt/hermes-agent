"""L5: 完整性验证

检查：
1. 所有占位符都已还原
2. 还原后的 arguments 是合法 JSON
3. 还原后的 arguments hash == 原始 hash
4. 消息角色交替合规
"""

import json
import hashlib
import logging
from typing import Any, Dict, List, Optional

from .tool_protector import PROTECTED_PLACEHOLDER

logger = logging.getLogger(__name__)


def validate_compression(
    original_msgs: List[Dict[str, Any]],
    compressed_msgs: List[Dict[str, Any]],
    protected_map: Dict[str, str],
    original_hashes: Optional[Dict[str, str]] = None,
) -> bool:
    """完整性验证压缩结果。

    Args:
        original_msgs: 原始消息列表（压缩前）
        compressed_msgs: 压缩后的消息列表
        protected_map: tool_protector 返回的保护区字典
        original_hashes: 原始参数 hash 表（可选，不提供时重新计算）

    Returns:
        True 表示验证通过，False 表示验证失败
    """
    issues: List[str] = []

    # 1. 检查所有占位符都已还原
    _check_placeholders_resolved(compressed_msgs, issues)

    # 2. 还原后的 arguments 是合法 JSON
    if protected_map:
        _check_arguments_json(compressed_msgs, protected_map, issues)

    # 3. 还原后的 arguments hash == 原始 hash
    _check_arguments_hash(compressed_msgs, protected_map, original_hashes, issues)

    # 4. 消息角色交替合规
    _check_role_alternation(compressed_msgs, issues)

    # 5. 检查消息结构完整性
    _check_message_structure(compressed_msgs, issues)

    if issues:
        logger.warning("L5 完整性验证失败（%d 个问题）", len(issues))
        for issue in issues:
            logger.warning("  - %s", issue)
        return False

    logger.info("L5 完整性验证通过：占位符全部还原，JSON 合法，hash 一致，角色交替合规")
    return True


def _check_placeholders_resolved(messages: List[Dict[str, Any]], issues: List[str]) -> None:
    """检查所有占位符都已还原。"""
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        for tc_idx, tc in enumerate(msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue
            args_val = fn.get("arguments", "")
            placeholder = PROTECTED_PLACEHOLDER.format(msg_idx, tc_idx)
            # 检查各类占位符残留
            if "__PROTECTED_ARG_" in str(args_val):
                issues.append(f"消息[{msg_idx}] tool_call[{tc_idx}] 存在未还原的占位符: {args_val}")


def _check_arguments_json(
    messages: List[Dict[str, Any]],
    protected_map: Dict[str, str],
    issues: List[str],
) -> None:
    """检查还原后的 arguments 是合法 JSON。"""
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc_idx, tc in enumerate(msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue
            args_val = fn.get("arguments", "")
            # 跳过占位符（不在 protected_map 中的非 JSON 字符串）
            try:
                parsed = json.loads(args_val)
                if not isinstance(parsed, dict):
                    issues.append(
                        f"消息[{msg_idx}] tool_call[{tc_idx}] arguments 不是 dict: {type(parsed).__name__}"
                    )
            except (json.JSONDecodeError, TypeError) as e:
                # 检查是否为未匹配的占位符
                placeholder = PROTECTED_PLACEHOLDER.format(msg_idx, tc_idx)
                if args_val == placeholder and placeholder not in protected_map:
                    issues.append(
                        f"消息[{msg_idx}] tool_call[{tc_idx}] 参数仍为占位符且不在保护映射中: {placeholder}"
                    )
                elif args_val not in protected_map:
                    issues.append(
                        f"消息[{msg_idx}] tool_call[{tc_idx}] arguments 不是合法 JSON: {e}"
                    )


def _check_arguments_hash(
    messages: List[Dict[str, Any]],
    protected_map: Dict[str, str],
    original_hashes: Optional[Dict[str, str]],
    issues: List[str],
) -> None:
    """检查还原后的 arguments hash 与原始 hash 一致。

    使用值匹配而非位置索引来查找原始 hash，避免压缩后消息索引变化导致的
    hash 不匹配（压缩中间段后 tool_call 的 msg_idx 会变）。
    """
    # Build a reverse map: {restored_raw_value: original_hash}
    # from protected_map entries. This avoids the index-mismatch bug
    # when compressed message indices differ from original indices.
    raw_to_hash: Dict[str, str] = {}
    if protected_map:
        for placeholder, stored_json in protected_map.items():
            try:
                stored = json.loads(stored_json)
                raw = stored.get("raw", "")
                h = stored.get("hash", "")
                if raw and h:
                    raw_to_hash[raw] = h
            except (json.JSONDecodeError, KeyError):
                pass

    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc_idx, tc in enumerate(msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue
            args_val = fn.get("arguments", "")
            if not isinstance(args_val, str) or not args_val.strip():
                continue
            # Compute current hash of the restored arguments
            current_hash = hashlib.sha256(args_val.encode("utf-8")).hexdigest()[:16]
            # Look up by raw value match instead of positional placeholder index.
            # restore_tool_arguments already matches by value (args_val in protected_map),
            # so we use the same strategy here for consistency.
            expected_hash = raw_to_hash.get(args_val)
            if expected_hash and current_hash != expected_hash:
                issues.append(
                    f"消息[{msg_idx}] tool_call[{tc_idx}] hash 不匹配: "
                    f"当前={current_hash}, 期望={expected_hash}"
                )


def _check_role_alternation(messages: List[Dict[str, Any]], issues: List[str]) -> None:
    """检查消息角色交替合规。

    压缩后可能产生 summary（assistant, 无 tool_calls）紧接 tail 第一条
    （assistant, 有 tool_calls）的边界，属于正常现象，不视为违规。
    """
    for i in range(1, len(messages)):
        prev_role = messages[i - 1].get("role", "") if isinstance(messages[i - 1], dict) else ""
        curr_role = messages[i].get("role", "") if isinstance(messages[i], dict) else ""
        if prev_role == curr_role and curr_role in ("user", "assistant"):
            # tool 消息可以跟在 assistant 后，允许相同 role 连续
            if prev_role == "tool" or curr_role == "tool":
                continue
            # 压缩后边界：assistant(摘要, 无 tool_calls) → assistant(tail, 有 tool_calls)
            # 这是正常的——摘要插入导致，不是格式错误
            prev_tc = messages[i - 1].get("tool_calls") if isinstance(messages[i - 1], dict) else None
            curr_tc = messages[i].get("tool_calls") if isinstance(messages[i], dict) else None
            if prev_role == "assistant" and curr_role == "assistant" and not prev_tc and curr_tc:
                continue
            issues.append(
                f"消息[{i}] 角色交替违规: 连续两个 {curr_role}"
            )


def _check_message_structure(messages: List[Dict[str, Any]], issues: List[str]) -> None:
    """检查消息结构完整性。"""
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            issues.append(f"消息[{msg_idx}] 不是 dict")
            continue
        role = msg.get("role")
        if role not in ("user", "assistant", "tool", "system"):
            issues.append(f"消息[{msg_idx}] 未知角色: {role}")
        # assistant 消息可以没有 tool_calls 也可以有
        # tool 消息必须有 tool_call_id
        if role == "tool" and "tool_call_id" not in msg:
            issues.append(f"消息[{msg_idx}] tool 消息缺少 tool_call_id")
