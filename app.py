# app.py
import streamlit as st
import google.generativeai as genai
import json
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="GenAI Customer Support", page_icon="💬", layout="centered")

# Load Gemini API key from Streamlit secrets
genai.configure(api_key="AIzaSyCHtoqd0rHgFQ5CKjrFfsUJ0MywuKfxpL8")

CHAT_MODEL = "gemini-2.5-flash"   # Gemini model
DOCS_PATH = "kb_docs.jsonl"       # small reference file (light RAG)
MAX_CONTEXT_DOCS = 1              # Only 1 document hint (minimal RAG)
# ----------------------------------------

# ---------- Load Knowledge Base(RAG) ----------
def load_kb(kb_path=DOCS_PATH):
    if not os.path.exists(kb_path):
        return []
    docs = []
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                text = j.get("text") or j.get("content") or ""
                if text.strip():
                    docs.append(text.strip())
            except:
                pass
    return docs

docs = load_kb()

# ---------- Simple doc search (no FAISS) ----------
def get_relevant_doc(query):
    """
    Lightweight doc search: pick the doc with max keyword overlap.
    """
    if not docs:
        return ""
    query_words = set(query.lower().split())
    best_doc = ""
    best_overlap = 0
    for d in docs:
        words = set(d.lower().split())
        overlap = len(query_words & words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_doc = d
    return best_doc if best_overlap > 1 else ""


# ---------- Streamlit UI ----------
st.title("💬 GenAI Customer Support Assistant (Gemini)")
st.write("Ask your support question — powered by Gemini with light contextual search.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a polite, helpful customer support assistant."}
    ]

user_input = st.text_area("Your message", height=100, placeholder="e.g., My order hasn't arrived yet.")

col1, col2 = st.columns([1, 3])
if col1.button("Send", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Minimal RAG: add one relevant doc as background info
        context = get_relevant_doc(user_input)
        context_note = f"\n\nContext info:\n{context}" if context else ""

        # Build conversation history as prompt
        history = ""
        for msg in st.session_state.messages[1:]:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            history += f"{prefix}: {msg['content']}\n"

        prompt = f"""You are a professional customer support assistant. 
Answer clearly and courteously. If you are unsure, politely say you’ll check further.

Conversation:
{history}
{context_note}
Assistant:"""

        with st.spinner("Generating response..."):
            model = genai.GenerativeModel(CHAT_MODEL)
            response = model.generate_content(prompt)

        reply = response.text.strip()
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# ---------- Display chat ----------
st.markdown("---")
st.subheader("Conversation")
for msg in st.session_state.messages[1:]:
    role = "You" if msg["role"] == "user" else "Assistant"
    st.markdown(f"**{role}:** {msg['content']}")

st.markdown("---")
if st.button("Reset Chat"):
    st.session_state.messages = [st.session_state.messages[0]]
    st.experimental_rerun()
