import streamlit as st
import yt_dlp
import whisper
import os
import uuid
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="YouTube Video Q&A",
    layout="wide"
)

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
# DOWNLOAD VIDEO
# ==========================================

def download_youtube(url, out_dir="videos"):

    os.makedirs(out_dir, exist_ok=True)

    video_id = str(uuid.uuid4())

    output_path = os.path.join(out_dir, f"{video_id}.mp4")

    ydl_opts = {
        "format": "best[ext=mp4]",
        "outtmpl": output_path,
        "quiet": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "geo_bypass": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0"
        }
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_path

# ==========================================
# TRANSCRIBE VIDEO
# ==========================================

def transcribe(video_path):

    result = whisper_model.transcribe(video_path)

    chunks = []

    for seg in result["segments"]:

        chunks.append({
            "text": seg["text"],
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2)
        })

    return chunks

# ==========================================
# BUILD FAISS INDEX
# ==========================================

def build_faiss_index(chunks):

    texts = [c["text"] for c in chunks]

    embeddings = embedder.encode(texts)

    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(embeddings)

    return index

# ==========================================
# SEARCH
# ==========================================

def search(query, index, chunks, k=5):

    q_emb = embedder.encode([query])

    q_emb = np.array(q_emb).astype("float32")

    distances, indices = index.search(q_emb, k)

    results = []

    for idx in indices[0]:

        if idx < len(chunks):

            results.append(chunks[idx])

    return results

# ==========================================
# FORMAT ANSWER
# ==========================================

def format_answer(results):

    response = ""

    for r in results:

        response += (
            f"⏱ {r['start']}s → {r['end']}s\n"
            f"{r['text']}\n\n"
        )

    return response

# ==========================================
# SESSION STATE
# ==========================================

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

# ==========================================
# UI
# ==========================================

st.title("🎥 YouTube Video Q&A System")

st.markdown(
    "Ask questions from any YouTube video using Whisper + FAISS."
)

url = st.text_input("Enter YouTube Video URL")

# ==========================================
# PROCESS VIDEO
# ==========================================

if st.button("Process Video"):

    if not url:

        st.warning("Please enter a YouTube URL.")

    else:

        try:

            with st.spinner("📥 Downloading video..."):
                video_path = download_youtube(url)

            st.success("Video downloaded successfully!")

            with st.spinner("🧠 Transcribing video with Whisper..."):
                chunks = transcribe(video_path)

            st.success("Transcription completed!")

            with st.spinner("📦 Building vector index..."):
                index = build_faiss_index(chunks)

            st.session_state.chunks = chunks
            st.session_state.index = index

            st.success("✅ Video processed successfully!")

        except Exception as e:

            st.error(f"Error: {str(e)}")

# ==========================================
# QUESTION ANSWERING
# ==========================================

if st.session_state.chunks is not None:

    st.subheader("❓ Ask Questions About the Video")

    question = st.text_input("Enter your question")

    if st.button("Ask Question"):

        if not question:

            st.warning("Please enter a question.")

        else:

            with st.spinner("🔍 Searching relevant segments..."):

                results = search(
                    question,
                    st.session_state.index,
                    st.session_state.chunks
                )

                answer = format_answer(results)

            st.subheader("📌 Relevant Answers")

            st.text(answer)

            st.subheader("🎯 Top Matching Segments")

            for r in results:

                st.markdown(
                    f"""
                    **⏱ {r['start']}s → {r['end']}s**

                    {r['text']}
                    """
                )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit + Whisper + FAISS")
