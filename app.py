import streamlit as st
import yt_dlp
import whisper
import os
import uuid
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return whisper_model, embedder

whisper_model, embedder = load_models()

# ==========================================
# UTILITIES
# ==========================================

def download_youtube(url, out_dir="videos"):
    os.makedirs(out_dir, exist_ok=True)
    video_id = str(uuid.uuid4())
    output_path = f"{out_dir}/{video_id}.mp4"

    ydl_opts = {
        "format": "mp4",
        "outtmpl": output_path,
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_path


def transcribe(video_path):
    result = whisper_model.transcribe(video_path)

    chunks = []

    for seg in result["segments"]:
        chunks.append({
            "text": seg["text"],
            "start": seg["start"],
            "end": seg["end"]
        })

    return chunks


def build_faiss_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    return index, embeddings


def search(query, index, chunks, k=5):
    q_emb = embedder.encode([query]).astype("float32")

    distances, indices = index.search(q_emb, k)

    results = []
    for idx in indices[0]:
        results.append(chunks[idx])

    return results


def format_answer(query, results):
    context = "\n".join(
        [f"{r['start']}s - {r['end']}s: {r['text']}" for r in results]
    )

    return context

# ==========================================
# STREAMLIT UI
# ==========================================

st.set_page_config(page_title="YouTube Q&A System", layout="wide")

st.title("🎥 YouTube Video Q&A with Timestamps")

url = st.text_input("Enter YouTube Link")

if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.index = None

# ==========================================
# PROCESS VIDEO
# ==========================================

if st.button("Process Video") and url:

    with st.spinner("Downloading video..."):
        video_path = download_youtube(url)

    with st.spinner("Transcribing with Whisper..."):
        chunks = transcribe(video_path)

    with st.spinner("Building search index..."):
        index, _ = build_faiss_index(chunks)

    st.session_state.chunks = chunks
    st.session_state.index = index

    st.success("Video processed successfully!")

# ==========================================
# Q&A SECTION
# ==========================================

if st.session_state.chunks is not None:

    st.subheader("Ask Questions About the Video")

    question = st.text_input("Your Question")

    if st.button("Ask") and question:

        with st.spinner("Searching relevant parts..."):

            results = search(
                question,
                st.session_state.index,
                st.session_state.chunks
            )

            answer = format_answer(question, results)

        st.subheader("📌 Answer with Timestamps")

        st.write(answer)

        st.subheader("🎯 Top Relevant Segments")

        for r in results:
            st.markdown(f"""
            **{r['start']}s → {r['end']}s**  
            {r['text']}
            """)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.markdown("Built with Whisper + FAISS + Streamlit")