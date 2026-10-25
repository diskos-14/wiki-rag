"""A simple chat UI for the wiki-rag backend.

Run (from frontend/):  uv run streamlit run app.py
The backend URL comes from API_BASE_URL (default http://localhost:8000); if the backend requires
an API key, set API_KEY and it's sent as the X-API-Key header.
"""

import os

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "").strip()
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}
TIMEOUT_SECONDS = 300  # the agent makes several model calls; Vertex AI can be slow

MODES = {
    "Agent (/agent/ask)": "/agent/ask",
    "Basic RAG (/ask)": "/ask",
}
EXAMPLES = [
    "What borders Uruguay?",
    "Who was born first, Isaac Newton or Blaise Pascal?",
    "Did Nikola Tesla work with Thomas Edison?",
    "Which country is larger, Uruguay or Finland?",
]

st.set_page_config(page_title="wiki-rag chat", page_icon="💬")


def new_chat() -> None:
    st.session_state.messages = []
    st.session_state.conversation_id = None


if "messages" not in st.session_state:
    new_chat()


# --- Sidebar ---------------------------------------------------------------------------------

with st.sidebar:
    st.header("wiki-rag")
    mode = st.radio(
        "Answer with",
        list(MODES),
        help="The agent handles comparisons and "
        "questions spanning several articles; basic RAG is faster.",
    )
    show_steps = st.toggle("Show agent steps", value=False)
    st.button("New chat", on_click=new_chat, use_container_width=True)

    st.subheader("Try")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example

    st.caption(f"Backend: {API_BASE_URL}")


# --- Rendering -------------------------------------------------------------------------------


def render_assistant(message: dict) -> None:
    st.markdown(message["content"])
    meta = message.get("meta", {})
    notes = []
    if meta.get("partial"):
        notes.append("🟡 partial answer")
    elif meta.get("answered") is False:
        notes.append("🚫 not answered" + (f" ({meta['reason']})" if meta.get("reason") else ""))
    if meta.get("seconds") is not None:
        notes.append(f"{meta['seconds']:.1f} s")
    if meta.get("endpoint"):
        notes.append(meta["endpoint"])
    if notes:
        st.caption(" · ".join(notes))

    cited = [s for s in message.get("sources", []) if s.get("cited")]
    if cited:
        with st.expander(f"Sources ({len(cited)})"):
            for source in cited:
                st.markdown(f"**[{source['number']}] {source['title']}**")
                st.caption(source["snippet"])

    if message.get("steps"):
        with st.expander("Agent steps"):
            for step in message["steps"]:
                st.markdown(f"`{step['node']}` · {step['ms'] / 1000:.1f} s  \n{step['summary']}")


st.title("💬 Ask Wikipedia")
st.caption("Answers come only from 41 English Wikipedia articles, with sources.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant(message)
        else:
            st.markdown(message["content"])


# --- Asking ----------------------------------------------------------------------------------


def ask(question: str) -> dict:
    endpoint = MODES[mode]
    payload = {"question": question, "debug": show_steps}
    if st.session_state.conversation_id:
        payload["conversation_id"] = st.session_state.conversation_id
    response = httpx.post(
        f"{API_BASE_URL}{endpoint}", json=payload, headers=HEADERS, timeout=TIMEOUT_SECONDS
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"{response.status_code}: {detail}")
    body = response.json()
    st.session_state.conversation_id = body["conversation_id"]
    return {
        "role": "assistant",
        "content": body["answer"],
        "sources": body.get("sources", []),
        "steps": body.get("steps"),
        "meta": {
            "answered": body.get("answered"),
            "partial": body.get("partial"),
            "reason": body.get("refusal_reason"),
            "seconds": body.get("latency_ms", 0) / 1000,
            "endpoint": endpoint,
        },
    }


question = st.chat_input("Ask a question…") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                reply = ask(question)
        except httpx.TimeoutException:
            st.error("The backend took too long. Please try again.")
        except (httpx.HTTPError, RuntimeError) as exc:
            st.error(f"Request failed: {exc}")
        else:
            render_assistant(reply)
            st.session_state.messages.append(reply)
