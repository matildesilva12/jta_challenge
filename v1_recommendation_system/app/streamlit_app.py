# Visual interface for the agentic recommendation system

import os
import sys
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent import run_agent
from data_loader import load_all

st.set_page_config(page_title="Sistema de Recomendação - Nintendo Switch", page_icon="🎮", layout="centered")


# Data loading (cached to avoid re-reading on every interaction)
def get_catalog():
    products, _ = load_all()
    return products


products = get_catalog()


LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/5/5d/Nintendo_Switch_Logo.svg"
st.markdown(
    f"""
    <div style="text-align:center; margin-bottom:0.5rem;">
        <div style="display:inline-block; background:#ffffff; padding:18px 30px;
                    border-radius:16px;">
            <img src="{LOGO_URL}" width="240" alt="Nintendo Switch"/>
        </div>
    </div>
    <h1 style="text-align:center; margin-top:0.5rem;">Recomendações Nintendo Switch</h1>
    """,
    unsafe_allow_html=True,
)

# Initialize the session history (isolated per user/browser)
if "history" not in st.session_state:
    st.session_state.history = []

# Button to reset the conversation
if st.button("🔄 Nova conversa"):
    st.session_state.history = []
    st.rerun()


if "OPENAI_API_KEY" not in os.environ:
    st.error(
        "A variável OPENAI_API_KEY não está definida. Define-a antes de correr:\n\n"
        "`export OPENAI_API_KEY=sk-...` e depois `streamlit run streamlit_app.py`"
    )
    st.stop()


# Chat input box
query = st.chat_input("Escreve o teu pedido...")

# Show the previous messages from the conversation
for hist_idx, msg in enumerate(st.session_state.history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if query:
   
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.history
    ]

    # Add user's question to the conversation
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.history.append({"role": "user", "content": query})

    # Run the agent and show the answer
    with st.chat_message("assistant"):
        with st.spinner("O agente está a pensar..."):
            try:
                answer = run_agent(query, history=history, verbose=False)
            except Exception as e:
                answer = f"Ocorreu um erro ao processar: {e}"

        st.markdown(answer)

    st.session_state.history.append(
        {"role": "assistant", "content": answer}
    )
