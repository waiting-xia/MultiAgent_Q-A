import requests
from datetime import datetime, timedelta
from langchain_core.tools import tool
from vector_store import hybrid_search


@tool
def rag_search_tool(query: str) -> str:
    """当遇到公司内部文档、特定知识库的问题时，使用此工具进行混合语义检索（BM25 + 向量）。"""
    results = hybrid_search(query, k=2)
    if results:
        context = "\n".join(results)
        return f"检索到的相关知识库内容：\n{context}"
    return "在知识库中未找到相关内容。"


@tool
def github_trending_tool(query: str) -> str:
    """获取 GitHub 上最近一周 star 数最多的 AI 相关开源项目 Top 10。
    参数 query 可为任意描述，实际返回固定的热门 AI 项目列表。"""
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    url = "https://api.github.com/search/repositories"
    params = {
        "q": f"(topic:ai OR topic:machine-learning OR topic:deep-learning OR topic:llm) created:>{one_week_ago}",
        "sort": "stars",
        "order": "desc",
        "per_page": 10,
    }
    headers = {"Accept": "application/vnd.github.v3+json"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        if not items:
            return "未找到最近一周的热门 AI 项目。"

        lines = [f"GitHub 最近一周热门 AI 项目 Top 10（查询时间: {datetime.now().strftime('%Y-%m-%d')}）:\n"]
        for i, repo in enumerate(items, 1):
            name = repo.get("full_name", "N/A")
            stars = repo.get("stargazers_count", 0)
            desc = repo.get("description", "无描述")
            html_url = repo.get("html_url", "")
            lines.append(f"{i}. **{name}** ⭐ {stars}")
            lines.append(f"   描述: {desc}")
            lines.append(f"   链接: {html_url}")
            lines.append("")

        return "\n".join(lines)
    except requests.RequestException as e:
        return f"GitHub API 请求失败: {str(e)}"


tools = [rag_search_tool, github_trending_tool]
