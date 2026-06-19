import streamlit as st
import tempfile
import os
from rag_pipeline import process_pdf_and_query

st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 DocuMind")
st.markdown("**AI-powered Document Intelligence** — Upload a PDF and ask anything!")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_ready" not in st.session_state:
    st.session_state.qa_ready = False
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

st.subheader("📄 Step 1: Upload your PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        st.session_state.pdf_path = tmp.name
    st.success(f"✅ '{uploaded_file.name}' uploaded successfully!")
    st.session_state.qa_ready = True

st.divider()

st.subheader("💬 Step 2: Ask anything from your PDF")

if not st.session_state.qa_ready:
    st.info("👆 Please upload a PDF first!")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your document...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document..."):
                try:
                    result = process_pdf_and_query(
                        st.session_state.pdf_path,
                        question
                    )
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    with st.expander("📚 Source Pages"):
                        for i, doc in enumerate(sources):
                            page_num = doc.metadata.get('page', None)
                            if page_num is not None:
                                st.markdown(f"**Page {page_num + 1}:**")
                            else:
                                st.markdown(f"**Source {i+1}:**")
                            content = doc.page_content if doc.page_content else "No content"
                            st.caption(content[:300] + "..." if len(content) > 300 else content)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(f"Error: {str(e)}")