import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from graph import app_graph
from report import generate_report

# ==========================================
# FastAPI 应用
# ==========================================

app = FastAPI(title="LangGraph Subgraph & ReAct API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ReportRequest(BaseModel):
    thread_id: str
    message: str


# ==========================================
# 流式聊天端点
# ==========================================

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    async def event_generator():
        config = {"configurable": {"thread_id": req.thread_id}}
        user_input = HumanMessage(content=req.message)

        print(f"\n[Debug] 收到前端请求: '{req.message}' (Thread: {req.thread_id})")
        print("[Debug] 开始执行 LangGraph 推理...\n[Agent 输出]: ", end="", flush=True)

        try:
            async for msg, metadata in app_graph.astream(
                {"messages": [user_input], "tool_outputs": []},
                config=config,
                stream_mode="messages",
            ):
                if metadata.get("langgraph_node") == "summarizer":
                    content = msg.content
                    if content and isinstance(content, str):
                        print(content, end="", flush=True)
                        yield content

            print("\n\n[Debug] LangGraph 推理完成，流式输出结束。")

        except Exception as e:
            error_msg = f"\n[Backend Error]: {str(e)}"
            print(error_msg)
            yield error_msg

    return StreamingResponse(event_generator(), media_type="text/plain")


# ==========================================
# 报告生成端点
# ==========================================

@app.post("/report")
async def report_endpoint(req: ReportRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    user_input = HumanMessage(content=req.message)

    try:
        result = await app_graph.ainvoke(
            {"messages": [user_input], "tool_outputs": []},
            config=config,
        )

        summary = result["messages"][-1].content if result["messages"] else ""
        tool_outputs = result.get("tool_outputs", [])

        filepath = generate_report(req.message, summary, tool_outputs)
        filename = os.path.basename(filepath)

        return {
            "status": "success",
            "filename": filename,
            "download_url": f"/download/{filename}",
            "summary": summary[:500],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 报告下载端点
# ==========================================

@app.get("/download/{filename}")
async def download_report(filename: str):
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    filepath = os.path.join(reports_dir, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="报告文件不存在")

    return FileResponse(filepath, filename=filename, media_type="text/markdown")
