import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title("📄 PDF Chatbot - Ask Me Anything!")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()

    st.success("PDF loaded successfully!")

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    query = st.text_input("Ask your question:")

    if query:
        q_embed = model.encode([query])
        _, top_indices = index.search(np.array(q_embed), k=3)
        answers = [chunks[i] for i in top_indices[0]]

        st.subheader("Answer:")
        st.write(answers[0])  # Best answer

        with st.expander("Other relevant answers"):
            for ans in answers[1:]:
                st.write(ans)
