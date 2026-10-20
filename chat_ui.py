import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["ZAI_API_KEY"] = "你的智谱API_Key填在这里"

# ==========================================
# 1. 加载与配置
# ==========================================
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

vectorstore = load_vectorstore()

# 提示词
template = """你是一个专业的简历助手，负责回答关于候选人简历的问题。
请严格根据以下简历片段回答问题，不要编造信息。
如果简历中没有相关信息，请直接说“简历中未提及”。

简历片段：
{context}

问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 大模型
llm = ChatOpenAI(
    model="glm-4.7-flash",
    openai_api_key=os.environ["ZAI_API_KEY"],
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ==========================================
# 2. Streamlit 界面与侧边栏
# ==========================================
st.set_page_config(page_title="多版本 AI 简历", page_icon="🧑‍💻", layout="wide")

# 初始化 session_state，用来临时存放 JD 分析报告
if "jd_report" not in st.session_state:
    st.session_state.jd_report = None

with st.sidebar:
    st.title("⚙️ 设置")
    
    # 选择简历版本
    selected_resume = st.selectbox("简历版本", ("测试岗位", "开发岗位"))
    
    st.markdown("---")
    
    # ⭐ JD 匹配度分析操作区（仅保留输入和按钮）
    st.markdown("### 🎯 JD 匹配度分析")
    st.caption("粘贴目标岗位的 JD，AI 将评估你的简历匹配度。")
    jd_text = st.text_area("目标岗位 JD", height=150, placeholder="在这里粘贴招聘要求...")
    
    if st.button("🚀 开始分析匹配度", use_container_width=True):
        if not jd_text.strip():
            st.warning("请先粘贴一段 JD 哦！")
        else:
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 4, "filter": {"resume_type": selected_resume}}
            )
            
            with st.spinner("正在对比简历与 JD..."):
                docs = retriever.invoke(jd_text)
                context = "\n\n".join(doc.page_content for doc in docs)
                
                analysis_prompt = f"""
                你是一位资深的 HR 专家和技术面试官。
                你的任务是评估候选人的简历与目标岗位 JD 的匹配程度。
                
                【目标岗位 JD】：
                {jd_text}
                
                【候选人的简历片段】：
                {context}
                
                请严格根据上述信息，输出一份结构化的匹配度分析报告，包含以下三个部分：
                1. 总体匹配度打分（满分 100 分，并给出简短理由）
                2. 核心优势（候选人具备的哪些技能/经验非常契合该岗位）
                3. 潜在差距与建议（候选人缺乏哪些关键技能，建议如何在面试中弥补或后续学习）
                
                请用清晰的 Markdown 格式输出，语言要专业、客观。
                """
                
                # 调用大模型，并将结果存入 session_state
                response = llm.invoke(analysis_prompt)
                st.session_state.jd_report = response.content
                st.success("分析完成！请在右侧查看报告。")

    st.markdown("---")
    
    if st.button("🗑️ 清空当前对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.jd_report = None # 清空对话时顺便清空报告
        st.rerun()


# ==========================================
# 3. 主界面聊天与报告展示
# ==========================================
st.title(f"📄 {selected_resume} - AI 简历机器人")
st.caption("你可以直接在下方提问，或者使用左侧的 JD 分析功能。")
st.divider()

# ⭐ 核心改动：如果 session_state 里有报告，就在主界面展示它！
if st.session_state.jd_report:
    st.markdown("### 📊 匹配度分析报告")
    # 加个按钮方便关闭报告，回到聊天模式
    if st.button("❌ 关闭报告，返回聊天", type="primary"):
        st.session_state.jd_report = None
        st.rerun()
    
    st.info(st.session_state.jd_report) # 用 info 或 markdown 展示报告
    st.divider()


# ----- 下面的聊天逻辑保持不变 -----
if "messages" not in st.session_state or st.session_state.get("current_resume") != selected_resume:
    st.session_state.messages = []
    st.session_state.current_resume = selected_resume

for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if question := st.chat_input(f"向 {selected_resume} 提问..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3, "filter": {"resume_type": selected_resume}}
    )
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    with st.chat_message("assistant", avatar="🤖"):
        with st.status(f"正在检索 {selected_resume} 简历...", expanded=False) as status:
            response = rag_chain.invoke(question)
            status.update(label="回答完成！", state="complete", expanded=False)
        st.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})