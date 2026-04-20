import os
import operator
from typing import Annotated, Sequence, TypedDict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import chromadb
from chromadb.utils import embedding_functions
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from langchain.agents import create_agent
from langchain_community.llms import Ollama
# ==========================================
# 1. 资源初始化与自定义工具 (RAG & MCP)
# ==========================================

# 初始化 ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="company_knowledge", 
    embedding_function=default_ef
)

if collection.count() == 0:
    collection.add(
        documents=[
            "员工每年有 15 天带薪年假，病假无限期，且支持远程办公。",
            "项目X目前处于UAT测试阶段，预计下个月上线，目前核心Bug已修复。",
            "公司差旅报销标准：国内每日补助 200 元，住宿上限 500 元。"
        ],
        metadatas=[{"type": "HR"}, {"type": "Project"}, {"type": "Finance"}],
        ids=["doc_001", "doc_002", "doc_003"]
    )

@tool
def rag_search_tool(query: str) -> str:
    """当遇到公司内部文档、特定知识库的问题时，使用此工具进行语义检索。"""
    results = collection.query(query_texts=[query], n_results=2)
    if results['documents'] and results['documents'][0]:
        context = "\n".join(results['documents'][0])
        return f"检索到的相关知识库内容：\n{context}"
    return "在知识库中未找到相关内容。"

@tool
def mcp_client_tool(resource_uri: str) -> str:
    """通过 Model Context Protocol 读取外部系统的资源。例如 Github 或 SQLite。"""
    if "github" in resource_uri.lower():
        return "MCP 读取到 Github 资源: 最新 commit 是修复了 API 路由 bug。"
    if "sqlite" in resource_uri.lower():
        return "MCP 读取到 SQLite 数据: 当前活跃用户数为 1024 人。"
    return f"MCP 服务成功访问资源: {resource_uri}，但内容为空。"

tools = [rag_search_tool, mcp_client_tool]

# ==========================================
# 2. 构建 ReAct 子图 (Subgraph)
# ==========================================

# 初始化 LLM
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# llm = Ollama(
#     model="qwen3.5-flas",  # 本地的模型名字
#     temperature=0.7,   # 随机性
#     base_url="http://localhost:11434"  # 默认地址
# )
llm = ChatOpenAI(model="qwen3.5-flash", 
               temperature=0.7,  
               base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
               api_key="", streaming=True)

# 使用 prebuilt 方法快速创建一个标准的 ReAct Agent 作为一个独立的图
# 这个子图内部自带了 Thought -> Action(Tool) -> Observation 的循环逻辑
react_system_prompt = """你是一个具备深度推理能力的研究员 Agent。
请根据用户的需求，自主思考并决定是否需要调用工具。
如果需要，反复调用 RAG 或 MCP 工具直到获取足够的上下文，然后输出你的阶段性研究结论。"""

react_research_subgraph = create_agent(
    model=llm,
    tools=tools,
    system_prompt=react_system_prompt
)

# ==========================================
# 3. 构建 父图 (Parent Graph)
# ==========================================

# 父图的状态管理
class ParentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 父图节点：调用 ReAct 子图
async def react_research_node(state: ParentState):
    """
    此节点充当桥梁，将父图的状态传递给子图，并提取子图的最终结果。
    """
    # 触发子图执行，传入当前的对话历史
    subgraph_result = await react_research_subgraph.ainvoke({"messages": state["messages"]})
    
    # 子图的内部状态包含了大量中间的 ToolCalls 和 ToolMessages (即 Thought/Observation)
    # 为了保持父图的主线对话清爽，我们通常只把子图输出的最后一条结论性消息合并回父图
    final_output = subgraph_result["messages"][-1]
    
    return {"messages": [final_output]}

# 父图节点：最终总结润色
async def summarizer_node(state: ParentState):
    messages = state["messages"]
    sys_msg = SystemMessage(content="你是面向客户的发言人。请基于研究员 Agent 提供的信息，输出最终的优质回答。要求结构清晰、语气专业。")
    response = await llm.ainvoke([sys_msg] + messages)
    return {"messages": [response]}

# 组装父图
parent_workflow = StateGraph(ParentState)

# 将子图的包装函数作为节点加入
parent_workflow.add_node("react_researcher", react_research_node)
parent_workflow.add_node("summarizer", summarizer_node)

# 定义父图的单一线性流程
parent_workflow.add_edge(START, "react_researcher")
parent_workflow.add_edge("react_researcher", "summarizer")
parent_workflow.add_edge("summarizer", END)

# 编译父图，挂载记忆模块
memory = MemorySaver()
app_graph = parent_workflow.compile(checkpointer=memory)

# ==========================================
# 4. 封装为 FastAPI 服务 (保持不变)
# ==========================================

app = FastAPI(title="LangGraph Subgraph & ReAct API")

# 2. 添加 CORS 中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源（仅供本地测试，生产环境请填写真实域名）
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法 (GET, POST, OPTIONS 等)
    allow_headers=["*"],  # 允许所有请求头
)


class ChatRequest(BaseModel):
    thread_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    async def event_generator():
        config = {"configurable": {"thread_id": req.thread_id}}
        user_input = HumanMessage(content=req.message)
        
        # 调试信息 1：确认请求进来了
        print(f"\n[Debug] 收到前端请求: '{req.message}' (Thread: {req.thread_id})")
        print("[Debug] 开始执行 LangGraph 推理...\n[Agent 输出]: ", end="", flush=True)
        
        try:
            async for msg, metadata in app_graph.astream(
                {"messages": [user_input]}, 
                config=config, 
                stream_mode="messages"
            ):
                if metadata.get("langgraph_node") == "summarizer":
                    content = msg.content
                    if content and isinstance(content, str):
                        
                        # 核心调试点：把模型刚刚吐出来的 Token 打印在后端控制台上
                        # end="" 防止每次都换行，flush=True 强制立即输出到屏幕
                        print(content, end="", flush=True) 
                        
                        yield content
                        
            # 调试信息 2：确认流式输出彻底结束
            print("\n\n[Debug] LangGraph 推理完成，流式输出结束。")
                        
        except Exception as e:
            error_msg = f"\n[Backend Error]: {str(e)}"
            print(error_msg) # 在后端控制台打印错误
            yield error_msg

    return StreamingResponse(event_generator(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
