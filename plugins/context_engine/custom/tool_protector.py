"""L1: 工具参数保护

扫描所有消息中的 tool_calls[].function.arguments，
替换为不可压缩的占位符，压缩后还原。
"""

import json
import hashlib
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

PROTECTED_PLACEHOLDER = "__PROTECTED_ARG_{}_{}__"


def protect_tool_arguments(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """扫描所有消息中的 tool_calls[].function.arguments，替换为占位符。

    Args:
        messages: 原始消息列表

    Returns:
        (修改后的messages, 保护区字典)
        保护区字典 key 为占位符字符串，value 为原始 arguments JSON 字符串
    """
    protected_map: Dict[str, str] = {}
    modified = False
    result: List[Dict[str, Any]] = []

    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            result.append(msg)
            continue

        if msg.get("role") != "assistant":
            result.append(msg)
            continue

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            result.append(msg)
            continue

        new_tool_calls: List[Dict[str, Any]] = []
        for tc_idx, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                new_tool_calls.append(tc)
                continue

            fn = tc.get("function")
            if not isinstance(fn, dict):
                new_tool_calls.append(tc)
                continue

            raw_args = fn.get("arguments", "")
            if not isinstance(raw_args, str) or not raw_args.strip():
                new_tool_calls.append(tc)
                continue

            # Compute hash of original arguments for validation
            arg_hash = hashlib.sha256(raw_args.encode("utf-8")).hexdigest()[:16]

            # Create placeholder
            placeholder = PROTECTED_PLACEHOLDER.format(msg_idx, tc_idx)
            protected_map[placeholder] = json.dumps({
                "raw": raw_args,
                "hash": arg_hash,
            })

            # Replace arguments with short placeholder — safe from LLM summarization
            new_fn = {**fn, "arguments": placeholder}
            new_tc = {**tc, "function": new_fn}
            new_tool_calls.append(new_tc)
            modified = True

        if modified:
            result.append({**msg, "tool_calls": new_tool_calls})
        else:
            result.append(msg)

    if modified:
        logger.info("L1 参数保护：保护了 %d 个 tool_call arguments", len(protected_map))

    return result if modified else messages, protected_map


def restore_tool_arguments(messages: List[Dict[str, Any]], protected_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """把占位符还原为原始 arguments。

    Args:
        messages: 压缩后的消息列表（含占位符）
        protected_map: protect_tool_arguments 返回的保护区字典

    Returns:
        还原后的消息列表
    """
    if not protected_map:
        return messages

    modified = False
    result: List[Dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            result.append(msg)
            continue

        if msg.get("role") != "assistant":
            result.append(msg)
            continue

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            result.append(msg)
            continue

        new_tool_calls: List[Dict[str, Any]] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                new_tool_calls.append(tc)
                continue

            fn = tc.get("function")
            if not isinstance(fn, dict):
                new_tool_calls.append(tc)
                continue

            args_val = fn.get("arguments", "")
            if args_val in protected_map:
                try:
                    stored = json.loads(protected_map[args_val])
                    original_raw = stored["raw"]
                    new_fn = {**fn, "arguments": original_raw}
                    new_tc = {**tc, "function": new_fn}
                    new_tool_calls.append(new_tc)
                    modified = True
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("L1 参数还原失败（占位符 %s）: %s", args_val, e)
                    new_tool_calls.append(tc)
            else:
                new_tool_calls.append(tc)

        if modified:
            result.append({**msg, "tool_calls": new_tool_calls})
        else:
            result.append(msg)

    if modified:
        logger.info("L1 参数还原：还原了占位符中的 tool_call arguments")

    return result


def compute_arg_hash(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """为所有 tool_call arguments 计算 hash，用于完整性验证。

    Returns:
        {占位符: hash值}
    """
    hashes: Dict[str, str] = {}
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc_idx, tc in enumerate(msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue
            raw_args = fn.get("arguments", "")
            if not isinstance(raw_args, str) or not raw_args.strip():
                continue
            placeholder = PROTECTED_PLACEHOLDER.format(msg_idx, tc_idx)
            arg_hash = hashlib.sha256(raw_args.encode("utf-8")).hexdigest()[:16]
            hashes[placeholder] = arg_hash
    return hashes
