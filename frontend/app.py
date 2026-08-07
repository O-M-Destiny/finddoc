# streamlit_app.py
import json
import uuid

import requests
import streamlit as st

API_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000/chat/stream")

st.set_page_config(page_title="FindDoc — NVIDIA Report Q&A", page_icon="📊")
st.title("FindDoc")
st.caption("Ask questions about NVIDIA's 2025 Annual Report")

#  session_id: created once, persists across reruns
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

#  chat history for DISPLAY only (the real memory lives in Redis on the backend)
if "messages" not in st.session_state:
    st.session_state.messages = []

#  render existing messages 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# handle new input
if question := st.chat_input("Ask a question about the report..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        queries_placeholder = st.empty()
        answer_placeholder = st.empty()
        full_answer = ""

        status_placeholder.caption(" Waking up the backend — this can take up to a minute if it's been idle...")

        try:
            response = requests.post(
                API_URL,
                json={"question": question, "session_id": st.session_state.session_id},
                stream=True,
                timeout=90,
            )
            response.raise_for_status()
            status_placeholder.empty()  # backend responded, clear the wake-up notice

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue

                event = json.loads(line[len("data: "):])

                if event["type"] == "queries":
                    queries_str = ", ".join(event["content"])
                    queries_placeholder.caption(f"🔍 Searching for: {queries_str}")

                elif event["type"] == "answer_token":
                    full_answer += event["content"]
                    answer_placeholder.markdown(full_answer + "▌")

                elif event["type"] == "error":
                    full_answer = event["content"]
                    answer_placeholder.markdown(full_answer)

            answer_placeholder.markdown(full_answer)
            queries_placeholder.empty()

        except requests.exceptions.Timeout:
            status_placeholder.empty()
            full_answer = "The backend took too long to respond — it may still be waking up. Please try asking again in a moment."
            answer_placeholder.markdown(full_answer)

        except requests.exceptions.ConnectionError:
            status_placeholder.empty()
            full_answer = "Couldn't reach the backend — is the FastAPI server running?"
            answer_placeholder.markdown(full_answer)

        except requests.exceptions.HTTPError as e:
            status_placeholder.empty()
            full_answer = f"Backend returned an error: {e}"
            answer_placeholder.markdown(full_answer)

    st.session_state.messages.append({"role": "assistant", "content": full_answer})