import os
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from tools import tools

# ==========================================
# LLM 初始化
# ==========================================

api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-07d3ec80b0b748cb9f8abeadfcda6d83")

llm = ChatOpenAI(
    model="qwen3.5-flash",
    temperature=0.7,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    streaming=True,
)

# ==========================================
# ReAct 子图
# ==========================================

react_system_prompt = """你是一个具备深度推理能力的研究员 Agent。
请根据用户的需求，自主思考并决定是否需要调用工具。
如果需要，反复调用 RAG 或 MCP 工具直到获取足够的上下文，然后输出你的阶段性研究结论。"""

react_research_subgraph = create_agent(
    model=llm,
    tools=tools,
    system_prompt=react_system_prompt,
)

# ==========================================
# 父图
# ==========================================


class ParentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    tool_outputs: Annotated[Sequence[str], operator.add]


async def react_research_node(state: ParentState):
    """调用 ReAct 子图，提取最终消息和工具输出。"""
    subgraph_result = await react_research_subgraph.ainvoke({"messages": state["messages"]})

    final_output = subgraph_result["messages"][-1]

    # 收集子图中的工具输出（ToolMessage）
    tool_outputs = []
    for msg in subgraph_result["messages"]:
        if hasattr(msg, "type") and msg.type == "tool":
            tool_outputs.append(str(msg.content))

    return {"messages": [final_output], "tool_outputs": tool_outputs}


async def summarizer_node(state: ParentState):
    """面向客户的发言人，润色研究员的输出。"""
    messages = state["messages"]
    sys_msg = SystemMessage(
        content="你是面向客户的发言人。请基于研究员 Agent 提供的信息，输出最终的优质回答。要求结构清晰、语气专业。"
    )
    response = await llm.ainvoke([sys_msg] + messages)
    return {"messages": [response], "tool_outputs": []}


# 组装父图
parent_workflow = StateGraph(ParentState)
parent_workflow.add_node("react_researcher", react_research_node)
parent_workflow.add_node("summarizer", summarizer_node)
parent_workflow.add_edge(START, "react_researcher")
parent_workflow.add_edge("react_researcher", "summarizer")
parent_workflow.add_edge("summarizer", END)

memory = MemorySaver()
app_graph = parent_workflow.compile(checkpointer=memory)
