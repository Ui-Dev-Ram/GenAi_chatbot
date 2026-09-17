from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver


try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "SERPER_API_KEY" in st.secrets:
        os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
except FileNotFoundError:
    pass  # no secrets.toml — fine locally, .env already loaded above

# ---- Page config 
st.set_page_config(
    page_title="Created by Ram — QnA at Groq speed",
    page_icon="⚡",
    layout="centered",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg: #12141A;
        --bg-panel: #181B22;
        --border: #262A34;
        --text: #E8E9ED;
        --text-dim: #8A8F9C;
        --accent: #F5A623;
    }

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    section[data-testid="stSidebar"] {
        background: var(--bg-panel);
        border-right: 1px solid var(--border);
    }

    .bolt-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0;
    }
    .bolt-subtitle {
        color: var(--text-dim);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        margin-top: 4px;
        margin-bottom: 1.5rem;
    }

    [data-testid="stChatMessage"] {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 4px 8px;
        margin-bottom: 10px;
    }

    a { color: var(--accent); }
    
    hr { border-color: var(--border); }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_agent():
    llm = ChatGroq(model="openai/gpt-oss-20b", streaming=True)
    search = GoogleSerperAPIWrapper()
    tools = [search.run]
    memory = MemorySaver()
    agent = create_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        system_prompt="You are Bolt, a fast, sharp, and helpful AI agent.",
    )
    return agent

agent = get_agent()

if "history" not in st.session_state:
    st.session_state.history = []

# ---- Sidebar
with st.sidebar:
    st.markdown("### ⚡ Created by Ram")
    st.caption("Groq LPU inference · LangChain agent · live web search")
    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("- `openai/gpt-oss-20b` on Groq\n- LangChain `create_agent`\n- Google Serper search tool")
    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()

# ---- Header
st.markdown('<div class="bolt-title">What\'s on the agenda today?</div>', unsafe_allow_html=True)
st.markdown('<div class="bolt-subtitle">answers at groq speed ⚡</div>', unsafe_allow_html=True)

# ---- Render existing history
for message in st.session_state.history:
    st.chat_message(message["role"]).markdown(message["content"])

# ---- Handle new input
query = st.chat_input("What's on your mind?")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("ai"):
        with st.spinner("Thum‑ja‑lo‑d‑ho‑ra‑ha‑he!"):
            try:
                res = agent.stream(
                    {"messages": [{"role": "user", "content": query}]},
                    {"configurable": {"thread_id": "1"}},
                    stream_mode="messages",
                )
                space = st.empty()
                message = ""
                for chunk in res:
                    piece = chunk[0]
                    text = piece.content if hasattr(piece, "content") else str(piece)
                    if text:
                        message += text
                        space.markdown(message)
                st.session_state.history.append({"role": "ai", "content": message})
            except Exception as e:
                st.error(f"Something went wrong: {e}")