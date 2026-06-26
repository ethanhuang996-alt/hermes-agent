"""L3: CCR 可逆存储

使用 SQLite 存储，路径 ~/.hermes/ccr.db
存储被摘要替换的原始消息，支持按 (session_id, msg_id) 检索。
"""

import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# SQLite 连接池（线程本地）
_local = threading.local()


def _get_db_path() -> str:
    """返回 CCR 数据库路径。"""
    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return os.path.join(hermes_home, "ccr.db")


def _get_connection() -> sqlite3.Connection:
    """获取当前线程的 SQLite 连接。"""
    if not hasattr(_local, "conn") or _local.conn is None:
        db_path = _get_db_path()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _local.conn = conn
        _init_db(conn)
    return _local.conn


def _init_db(conn: sqlite3.Connection) -> None:
    """初始化数据库表。"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ccr_original (
            session_id TEXT NOT NULL,
            msg_id TEXT NOT NULL,
            message_json TEXT NOT NULL,
            stored_at REAL NOT NULL,
            compressed_msg_ids TEXT,
            PRIMARY KEY (session_id, msg_id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ccr_session
        ON ccr_original(session_id, stored_at)
    """)
    conn.commit()


class CcrStore:
    """CCR 存储管理器。

    使用 Hermes 已有的 SQLite 基础设施，路径 ~/.hermes/ccr.db。
    """

    @staticmethod
    def store_original(session_id: str, msg_id: str, original_message: Dict[str, Any],
                       compressed_msg_ids: Optional[List[str]] = None) -> None:
        """存储一条原始消息。

        Args:
            session_id: 会话 ID
            msg_id: 消息 ID（通常是索引或消息的唯一标识）
            original_message: 原始消息 dict
            compressed_msg_ids: 与本次压缩关联的其他 msg_id 列表（可选）
        """
        try:
            conn = _get_connection()
            conn.execute(
                """INSERT OR REPLACE INTO ccr_original
                   (session_id, msg_id, message_json, stored_at, compressed_msg_ids)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    session_id,
                    str(msg_id),
                    json.dumps(original_message, ensure_ascii=False),
                    time.time(),
                    json.dumps(compressed_msg_ids or [], ensure_ascii=False),
                ),
            )
            conn.commit()
        except Exception as e:
            logger.error("CCR 存储失败 (session=%s, msg=%s): %s", session_id, msg_id, e)

    @staticmethod
    def store_batch(session_id: str, messages: List[Dict[str, Any]],
                    msg_ids: Optional[List[str]] = None) -> List[str]:
        """批量存储原始消息。

        Args:
            session_id: 会话 ID
            messages: 原始消息列表
            msg_ids: 对应的消息 ID 列表（默认为索引字符串）

        Returns:
            使用的 msg_id 列表
        """
        ids = msg_ids or [str(i) for i in range(len(messages))]
        for msg_id, msg in zip(ids, messages):
            CcrStore.store_original(session_id, msg_id, msg)
        return ids

    @staticmethod
    def retrieve_original(session_id: str, msg_id: str) -> Optional[Dict[str, Any]]:
        """从 CCR 检索原始消息。

        Args:
            session_id: 会话 ID
            msg_id: 消息 ID

        Returns:
            原始消息 dict，未找到时返回 None
        """
        try:
            conn = _get_connection()
            row = conn.execute(
                "SELECT message_json FROM ccr_original WHERE session_id = ? AND msg_id = ?",
                (session_id, str(msg_id)),
            ).fetchone()
            if row:
                return json.loads(row["message_json"])
            return None
        except Exception as e:
            logger.error("CCR 检索失败 (session=%s, msg=%s): %s", session_id, msg_id, e)
            return None

    @staticmethod
    def retrieve_batch(session_id: str, msg_ids: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """批量检索原始消息。

        Args:
            session_id: 会话 ID
            msg_ids: 消息 ID 列表

        Returns:
            {msg_id: original_message_or_None}
        """
        results: Dict[str, Optional[Dict[str, Any]]] = {}
        for msg_id in msg_ids:
            results[msg_id] = CcrStore.retrieve_original(session_id, msg_id)
        return results

    @staticmethod
    def store_compression_record(session_id: str, compressed_msg_ids: List[str]) -> None:
        """存储一次压缩操作的记录，记录哪些 msg_id 被压缩了。

        Args:
            session_id: 会话 ID
            compressed_msg_ids: 本次压缩涉及的消息 ID 列表
        """
        try:
            conn = _get_connection()
            conn.execute(
                """INSERT INTO ccr_original
                   (session_id, msg_id, message_json, stored_at, compressed_msg_ids)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    session_id,
                    f"_compression_record_{int(time.time() * 1000)}",
                    json.dumps({"type": "compression_record"}, ensure_ascii=False),
                    time.time(),
                    json.dumps(compressed_msg_ids, ensure_ascii=False),
                ),
            )
            conn.commit()
        except Exception as e:
            logger.error("CCR 存储压缩记录失败 (session=%s): %s", session_id, e)

    @staticmethod
    def list_compressed_ids(session_id: str) -> List[str]:
        """列出指定 session 所有被压缩的消息 ID。

        Args:
            session_id: 会话 ID

        Returns:
            压缩记录中的 msg_id 列表
        """
        collected: List[str] = []
        try:
            conn = _get_connection()
            rows = conn.execute(
                "SELECT compressed_msg_ids FROM ccr_original "
                "WHERE session_id = ? AND compressed_msg_ids IS NOT NULL",
                (session_id,),
            ).fetchall()
            for row in rows:
                try:
                    ids = json.loads(row["compressed_msg_ids"])
                    if isinstance(ids, list):
                        collected.extend(ids)
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception as e:
            logger.error("CCR 列出压缩 ID 失败 (session=%s): %s", session_id, e)
        return list(dict.fromkeys(collected))  # deduplicate, preserve order

    @staticmethod
    def cleanup_old(ttl_hours: int = 24) -> int:
        """清理超过 TTL 的旧记录。

        Args:
            ttl_hours: 保留小时数

        Returns:
            删除的记录数
        """
        cutoff = time.time() - (ttl_hours * 3600)
        try:
            conn = _get_connection()
            cursor = conn.execute(
                "DELETE FROM ccr_original WHERE stored_at < ?",
                (cutoff,),
            )
            conn.commit()
            deleted = cursor.rowcount
            if deleted:
                logger.info("CCR 清理：删除了 %d 条超过 %d 小时的旧记录", deleted, ttl_hours)
            return deleted
        except Exception as e:
            logger.error("CCR 清理失败: %s", e)
            return 0

    @staticmethod
    def clear_session(session_id: str) -> int:
        """清空指定 session 的所有记录。

        Args:
            session_id: 会话 ID

        Returns:
            删除的记录数
        """
        try:
            conn = _get_connection()
            cursor = conn.execute(
                "DELETE FROM ccr_original WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
            deleted = cursor.rowcount
            if deleted:
                logger.info("CCR 清空 session %s：删除了 %d 条记录", session_id, deleted)
            return deleted
        except Exception as e:
            logger.error("CCR 清空 session 失败 (%s): %s", session_id, e)
            return 0

    @staticmethod
    def close() -> None:
        """关闭当前线程的数据库连接。"""
        if hasattr(_local, "conn") and _local.conn is not None:
            try:
                _local.conn.close()
            except Exception:
                pass
            _local.conn = None
