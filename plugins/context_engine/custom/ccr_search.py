"""CCR 搜索工具 — 按关键词搜索压缩掉的原始消息。

可搜所有 session 被压缩过的消息（跨 session 可用），也可指定 session_id 限定。
限制：只有触发过压缩的 session 才会有数据。从未触发压缩的 session 要用 session_search。

用法（agent 内部调用）：
  ccr_search("Nginx 端口") → 返回所有内容中包含 "Nginx 端口" 的原始消息
  ccr_search("Nginx 端口", session_id="20260625_194115_40641b")
  
工具注册后，通过 Hermes 工具层暴露给 agent。
"""

import json, sqlite3, re, os
from typing import Optional, List, Dict, Any

CCR_DB = r"G:\hermes\hermes-agent\plugins\context_engine\custom\ccr.db"

def _get_conn():
    path = CCR_DB
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # Ensure schema exists (safe to call multiple times)
    conn.execute("""CREATE TABLE IF NOT EXISTS ccr_original (
        session_id TEXT NOT NULL,
        msg_id TEXT NOT NULL,
        message_json TEXT NOT NULL,
        compressed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (session_id, msg_id)
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ccr_session ON ccr_original(session_id)")
    conn.commit()
    return conn


def ccr_search(keyword: str, session_id: Optional[str] = None, limit: int = 10) -> str:
    """搜索压缩过的原始消息。
    
    Args:
        keyword: 搜索关键词
        session_id: 可选，限定到某个 session
        limit: 最多返回条数
        
    Returns:
        格式化结果文本，包含匹配的消息内容
    """
    if not keyword:
        return "请提供搜索关键词"
    
    try:
        conn = _get_conn()
        
        # 构建查询
        like_pattern = f"%{keyword}%"
        if session_id:
            rows = conn.execute(
                "SELECT session_id, msg_id, message_json FROM ccr_original "
                "WHERE session_id = ? AND message_json LIKE ? LIMIT ?",
                (session_id, like_pattern, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT session_id, msg_id, message_json FROM ccr_original "
                "WHERE message_json LIKE ? LIMIT ?",
                (like_pattern, limit)
            ).fetchall()
        
        if not rows:
            return f"未找到包含「{keyword}」的原始消息"
        
        results = []
        for row in rows:
            try:
                msg = json.loads(row["message_json"])
            except:
                msg = {"content": row["message_json"][:200]}
            
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            content = str(content)[:300]
            
            results.append(
                f"[{row['session_id'][:20]} msg:{row['msg_id']}] "
                f"role={role}\n"
                f"  {content}\n"
            )
        
        header = f"找到 {len(rows)} 条包含「{keyword}」的原始消息:\n"
        return header + "\n".join(results)
    
    except Exception as e:
        return f"搜索失败: {e}"
    finally:
        try:
            conn.close()
        except:
            pass
