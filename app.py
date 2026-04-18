import uuid
import requests
import streamlit as st
import time

# ==========================================
# 1. 基础配置
# ==========================================
st.title("智扫通机器人智能客服")
st.divider()

# 配置你的 FastAPI 后端地址
FASTAPI_URL = "http://localhost:8000/chat"

# ==========================================
# 2. 初始化 Session State
# ==========================================
# 替代原本的 agent 实例，现在我们只需要一个 thread_id 传给后端即可维持记忆
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = "session_" + str(uuid.uuid4())[:8]

if "message" not in st.session_state:
    st.session_state["message"] = []

# ==========================================
# 3. 渲染历史消息
# ==========================================
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# ==========================================
# 4. 处理用户交互与调用后端
# ==========================================
prompt = st.chat_input("请输入您的问题...")

if prompt:
    # 显示用户消息并存入状态
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 创建一个生成器函数，用于向 FastAPI 发起请求并 yield 数据块
    def stream_from_fastapi():
        full_text = []  # 用于拼接完整的回复字符串
        
        try:
            with requests.post(
                FASTAPI_URL,
                json={"thread_id": st.session_state["thread_id"], "message": prompt},
                stream=True,
                timeout=60
            ) as response:
                response.raise_for_status()
                
                # 实时遍历后端吐出的 Token
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_text.append(chunk)
                        
                        # 借鉴你原代码的思路：增加微小的延迟，让打字机效果更平滑
                        for char in chunk:
                            time.sleep(0.01)
                            yield char
                            
        except Exception as e:
            error_msg = f"\n⚠️ 网络请求出错，请检查 FastAPI 服务: {str(e)}"
            full_text.append(error_msg)
            yield error_msg
            
        # 当生成器执行完毕（流式输出结束）时，将拼接好的完整字符串存入历史记录
        final_string = "".join(full_text)
        st.session_state["message"].append({"role": "assistant", "content": final_string})

    # 渲染 AI 消息容器
    with st.chat_message("assistant"):
        # st.write_stream 会自动拉取我们的 stream_from_fastapi 生成器
        st.write_stream(stream_from_fastapi())
        
    # 注意：这里不需要再调用 st.rerun() 了。
    # Streamlit 的 write_stream 结合 session_state.append 已经完美处理了页面状态。
    # 强制 rerun 反而会导致页面多闪烁一次。