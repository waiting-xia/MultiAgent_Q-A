import os
from datetime import datetime


def generate_report(query: str, summary: str, tool_outputs: list[str]) -> str:
    """生成 Markdown 报告并保存到 ./reports/ 目录，返回文件路径。"""
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.md"
    filepath = os.path.join(reports_dir, filename)

    lines = [
        f"# AI 研究报告",
        f"",
        f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"---",
        f"",
        f"## 查询问题",
        f"",
        f"{query}",
        f"",
        f"---",
        f"",
        f"## 研究总结",
        f"",
        f"{summary}",
        f"",
    ]

    github_section = []
    rag_section = []

    for output in tool_outputs:
        if "github" in output.lower() or "仓库" in output or "star" in output.lower():
            github_section.append(output)
        elif "检索" in output or "知识库" in output or "参考资料" in output:
            rag_section.append(output)

    if github_section:
        lines.extend([
            f"---",
            f"",
            f"## GitHub 热门 AI 项目",
            f"",
        ])
        for item in github_section:
            lines.append(f"{item}")
            lines.append(f"")

    if rag_section:
        lines.extend([
            f"---",
            f"",
            f"## RAG 检索来源",
            f"",
        ])
        for item in rag_section:
            lines.append(f"{item}")
            lines.append(f"")

    lines.extend([
        f"---",
        f"",
        f"*本报告由 MultiAgent 自动生成*",
    ])

    content = "\n".join(lines)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath
