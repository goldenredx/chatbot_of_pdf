import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------- Load Models ----------
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_generator():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    return tokenizer, model

embedder = load_embedder()
gen_tokenizer, gen_model = load_generator()

# ---------- Helper Functions ----------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=150, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def build_index(chunks):
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

def retrieve_chunks(query, chunks, index, top_k=3):
    query_emb = embedder.encode([query]).astype('float32')
    distances, indices = index.search(query_emb, top_k)
    retrieved = [chunks[i] for i in indices[0]]
    return retrieved

def generate_answer(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"Answer in a short, clear sentence using the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = gen_model.generate(**inputs, max_length=500)
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📚 RAG Chatbot — Chat with your Document")
st.write("Upload a PDF and ask questions about its content.")

if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.index = None

uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file and st.session_state.chunks is None:
    with st.spinner("Processing document..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(raw_text)
        index, embeddings = build_index(chunks)
        st.session_state.chunks = chunks
        st.session_state.index = index
    st.success(f"Document processed into {len(chunks)} chunks. Ready to chat!")

if st.session_state.chunks:
    query = st.text_input("Ask a question about the document")
    if st.button("Get Answer") and query.strip():
        with st.spinner("Retrieving and generating answer..."):
            retrieved = retrieve_chunks(query, st.session_state.chunks, st.session_state.index)
            answer = generate_answer(query, retrieved)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("View Retrieved Context Chunks"):
            for i, chunk in enumerate(retrieved):
                st.markdown(f"**Chunk {i+1}:**")
                st.text(chunk)