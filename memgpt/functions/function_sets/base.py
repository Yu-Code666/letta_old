from typing import Optional
import datetime  # 导入处理日期和时间的模块
import os  # 导入操作系统相关的模块（虽然本文件没用上，但保留以供未来扩展）
import json  # 导入用于处理JSON格式数据的模块
import math  # 导入数学相关模块

from ...constants import MAX_PAUSE_HEARTBEATS, RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE  # 导入项目中的常量

### Functions / tools the agent can use
# All functions should return a response string (or None)
# If the function fails, throw an exception

def send_message(self, message: str):
    """
    Sends a message to the human user.

    Args:
        message (str): Message contents. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    # 调用界面方法向用户发送消息
    self.interface.assistant_message(message)
    return None

# Construct the docstring dynamically (since it should use the external constants)
pause_heartbeats_docstring = f"""
Temporarily ignore timed heartbeats. You may still receive messages from manual heartbeats and other events.

Args:
    minutes (int): Number of minutes to ignore heartbeats for. Max value of {MAX_PAUSE_HEARTBEATS} minutes ({MAX_PAUSE_HEARTBEATS // 60} hours).

Returns:
    str: Function status response
"""

def pause_heartbeats(self, minutes: int):
    # 限制暂停心跳的最大时长
    minutes = min(MAX_PAUSE_HEARTBEATS, minutes)

    # Record the current time
    self.pause_heartbeats_start = datetime.datetime.now()  # 记录当前的起始时间
    # And record how long the pause should go for
    self.pause_heartbeats_minutes = int(minutes)  # 记录暂停心跳的分钟数

    return f"Pausing timed heartbeats for {minutes} min"

# 为 pause_heartbeats 设置动态文档字符串
pause_heartbeats.__doc__ = pause_heartbeats_docstring

def core_memory_append(self, name: str, content: str):
    """
    Append to the contents of core memory.

    Args:
        name (str): Section of the memory to be edited (persona or human).
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    # 追加内容到核心记忆的特定部分
    new_len = self.memory.edit_append(name, content)
    # 重建内存以应用更改
    self.rebuild_memory()
    return None

def core_memory_replace(self, name: str, old_content: str, new_content: str):
    """
    Replace to the contents of core memory. To delete memories, use an empty string for new_content.

    Args:
        name (str): Section of the memory to be edited (persona or human).
        old_content (str): String to replace. Must be an exact match.
        new_content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    # 调用 edit_replace 方法替换核心记忆内容，返回新的内容长度
    new_len = self.memory.edit_replace(name, old_content, new_content)
    # 调用 rebuild_memory 方法，重建内存以应用刚才的更改
    self.rebuild_memory()
    # 无返回值，始终返回 None
    return None

def conversation_search(self, query: str, page: Optional[int] = 0):
    """
    Search prior conversation history using case-insensitive string matching.

    Args:
        query (str): String to search for.
        page (int): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

    Returns:
        str: Query result string
    """
    count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE  # 每页搜索的数量
    # 在会话历史中进行文本检索
    results, total = self.persistence_manager.recall_memory.text_search(query, count=count, start=page * count)
    num_pages = math.ceil(total / count) - 1  # 计算总页数，0为起始下标
    if len(results) == 0:
        results_str = f"No results found."  # 没有命中结果
    else:
        # 格式化返回结果和分页信息
        results_pref = f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        results_formatted = [f"timestamp: {d['timestamp']}, {d['message']['role']} - {d['message']['content']}" for d in results]
        results_str = f"{results_pref} {json.dumps(results_formatted)}"
    return results_str

def conversation_search_date(self, start_date: str, end_date: str, page: Optional[int] = 0):
    """
    Search prior conversation history using a date range.

    Args:
        start_date (str): The start of the date range to search, in the format 'YYYY-MM-DD'.
        end_date (str): The end of the date range to search, in the format 'YYYY-MM-DD'.
        page (int): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

    Returns:
        str: Query result string
    """
    count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE  # 每页检索数量
    # 根据日期范围在会话历史中检索
    results, total = self.persistence_manager.recall_memory.date_search(start_date, end_date, count=count, start=page * count)
    num_pages = math.ceil(total / count) - 1  # 计算总页数（0起始索引）
    if len(results) == 0:
        results_str = f"No results found."
    else:
        # 格式化返回结果和分页信息
        results_pref = f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        results_formatted = [f"timestamp: {d['timestamp']}, {d['message']['role']} - {d['message']['content']}" for d in results]
        results_str = f"{results_pref} {json.dumps(results_formatted)}"
    return results_str

def archival_memory_insert(self, content: str):
    """
    Add to archival memory. Make sure to phrase the memory contents such that it can be easily queried later.

    Args:
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    # 插入内容到存档记忆（通常为长期可检索存储）
    self.persistence_manager.archival_memory.insert(content)
    return None

def archival_memory_search(self, query: str, page: Optional[int] = 0):
    """
    Search archival memory using semantic (embedding-based) search.

    Args:
        query (str): String to search for.
        page (Optional[int]): Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).

    Returns:
        str: Query result string
    """
    count = RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE  # 每页返回的数量
    # 执行存档记忆的语义搜索（通常是向量/嵌入检索）
    results, total = self.persistence_manager.archival_memory.search(query, count=count, start=page * count)
    num_pages = math.ceil(total / count) - 1  # 计算结果总页数（0起始）
    if len(results) == 0:
        results_str = f"No results found."
    else:
        # 格式化显示返回的存档记忆内容
        results_pref = f"Showing {len(results)} of {total} results (page {page}/{num_pages}):"
        results_formatted = [f"timestamp: {d['timestamp']}, memory: {d['content']}" for d in results]
        results_str = f"{results_pref} {json.dumps(results_formatted)}"
    return results_str
