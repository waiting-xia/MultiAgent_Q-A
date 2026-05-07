import uuid
import requests
import streamlit as st
import time

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(
    page_title="多智能体问答系统",
    page_icon="🤖",
    layout="wide",
)

# 自定义 CSS
st.markdown("""
<style>
    /* 全局字体和背景 */
    .stApp {
        background-color: #f0f2f6;
    }

    /* 标题区域 */
    .main-title {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }

    /* 聊天气泡 */
    .stChatMessage {
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    /* 侧边栏 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0e0;
    }

    /* 状态指示器 */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-online {
        background-color: #d4edda;
        color: #155724;
    }
    .status-offline {
        background-color: #f8d7da;
        color: #721c24;
    }

    /* 输入框美化 */
    .stChatInput textarea {
        border-radius: 12px;
        border: 2px solid #667eea;
        font-size: 1rem;
    }

    /* 按钮美化 */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* 分割线 */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 1rem 0;
    }

    /* 欢迎消息 */
    .welcome-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }
    .welcome-box h3 {
        color: #333;
        margin-bottom: 0.5rem;
    }
    .welcome-box p {
        color: #666;
        font-size: 0.9rem;
    }

    /* 功能卡片 */
    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #667eea;
    }
    .feature-card h4 {
        margin: 0 0 0.3rem 0;
        color: #333;
    }
    .feature-card p {
        margin: 0;
        color: #888;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    st.markdown("## ⚙️ 控制面板")
    st.divider()

    # 后端状态检测
    st.markdown("### 服务状态")
    try:
        resp = requests.get("http://localhost:8000/docs", timeout=2)
        if resp.status_code == 200:
            st.markdown('<span class="status-badge status-online">● 后端运行中</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-offline">● 后端异常</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<span class="status-badge status-offline">● 后端未连接</span>', unsafe_allow_html=True)
        st.caption("请先运行 `python app.py` 启动后端")

    st.divider()

    # 会话信息
    st.markdown("### 会话信息")
    st.code(st.session_state.get("thread_id", "未初始化"), language=None)

    st.divider()

    # 操作按钮
    st.markdown("### 操作")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state["message"] = []
            st.session_state["thread_id"] = "session_" + str(uuid.uuid4())[:8]
            st.rerun()
    with col2:
        if st.button("📋 新会话", use_container_width=True):
            st.session_state["thread_id"] = "session_" + str(uuid.uuid4())[:8]
            st.session_state["message"] = []
            st.rerun()

    st.divider()

    # 示例问题
    st.markdown("### 💡 示例问题")
    examples = [
        "公司年假政策是什么？",
        "项目X目前进展如何？",
        "差旅报销标准是多少？",
        "最近有什么热门AI项目？",
    ]
    for ex in examples:
        if st.button(f"📝 {ex}", use_container_width=True, key=f"ex_{ex}"):
            st.session_state["example_query"] = ex
            st.rerun()

    st.divider()
    st.caption("Powered by LangGraph + FastAPI")

# ==========================================
# 主区域标题
# ==========================================
st.markdown('<h1 class="main-title">🤖 多智能体问答系统</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于 LangGraph 的 ReAct Agent | 支持 RAG 检索 + GitHub 热门项目查询</p>', unsafe_allow_html=True)

# ==========================================
# 初始化 Session State
# ==========================================
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = "session_" + str(uuid.uuid4())[:8]

if "message" not in st.session_state:
    st.session_state["message"] = []

# ==========================================
# 欢迎区域（无消息时显示）
# ==========================================
if not st.session_state["message"]:
    st.markdown("""
    <div class="welcome-box">
        <h3>👋 你好！我是你的 AI 信息收集助手</h3>
        <p>我可以帮你查询内部知识库，也可以搜索 GitHub 热门 AI 项目。</p>
        <p>试试在左侧点击示例问题，或直接输入你的问题吧！</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="feature-card">
            <h4>📚 知识库检索</h4>
            <p>混合 BM25 + 向量检索，精准匹配公司内部文档</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="feature-card">
            <h4>🔥 GitHub 追踪</h4>
            <p>实时获取最近一周 GitHub 热门 AI 开源项目</p>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 研究报告</h4>
            <p>自动生成结构化的 Markdown 研究报告</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 渲染历史消息
# ==========================================
for msg in st.session_state["message"]:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# ==========================================
# 处理用户输入
# ==========================================
# 支持侧边栏示例问题触发
example_query = st.session_state.pop("example_query", None)
prompt = st.chat_input("请输入你的问题...") or example_query

if prompt:
    # 显示用户消息
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 流式调用后端
    def stream_from_fastapi():
        full_text = []
        try:
            with requests.post(
                "http://localhost:8000/chat",
                json={"thread_id": st.session_state["thread_id"], "message": prompt},
                stream=True,
                timeout=120,
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_text.append(chunk)
                        for char in chunk:
                            time.sleep(0.01)
                            yield char
        except requests.ConnectionError:
            error_msg = "\n\n⚠️ **无法连接后端服务**，请确认已运行 `python app.py` 启动 FastAPI 服务。"
            full_text.append(error_msg)
            yield error_msg
        except requests.Timeout:
            error_msg = "\n\n⚠️ **请求超时**，AI 推理时间过长，请稍后重试。"
            full_text.append(error_msg)
            yield error_msg
        except Exception as e:
            error_msg = f"\n\n⚠️ **请求出错**: {str(e)}"
            full_text.append(error_msg)
            yield error_msg

        final_string = "".join(full_text)
        st.session_state["message"].append({"role": "assistant", "content": final_string})

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🧠 Agent 正在思考并调用工具..."):
            st.write_stream(stream_from_fastapi())
