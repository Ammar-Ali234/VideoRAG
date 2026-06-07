import streamlit as st
import yt_dlp
import whisper
import os
import uuid
import faiss
import numpy as np
import imageio_ffmpeg
import shutil

from sentence_transformers import SentenceTransformer

os.environ["HF_HOME"] = "/tmp/huggingface"

os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

# ==========================================
# PAGE CONFIG & STYLING
# ==========================================

st.set_page_config(
    page_title="YouTube Video Q&A",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS for a premium look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stTextInput > div > div > input {
        background-color: #1a1c24;
        color: #ffffff;
        border-radius: 10px;
        border: 1px solid #3e424b;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #ff2b2b;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
        transform: translateY(-2px);
    }
    .status-card {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    h1 {
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .stMarkdown p {
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("tiny")
    embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
    return whisper_model, embedder

whisper_model, embedder = load_models()

# ==========================================
# DOWNLOAD VIDEO
# ==========================================

def download_youtube(url, out_dir="videos"):

    os.makedirs(out_dir, exist_ok=True)

    video_id = str(uuid.uuid4())
    output_path = os.path.join(out_dir, video_id)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_path}.%(ext)s",
        "quiet": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "geo_bypass": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0"
        },
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    final_path = f"{output_path}.mp3"
    return final_path

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

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        "### Ask questions from any YouTube video using Whisper + FAISS."
    )
    url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

with col2:
    st.write("##") # Spacing
    process_button = st.button("🚀 Process Video")

# ==========================================
# PROCESS VIDEO
# ==========================================

if process_button:

    if not url:

        st.warning("Please enter a YouTube URL.")

    else:

        try:

            with st.spinner("📥 Downloading video..."):
                video_path = download_youtube(url)

            st.success("Video downloaded successfully!")

            with st.spinner("🧠 Transcribing video with Whisper..."):
                chunks = transcribe(video_path)

            # Cleanup: Remove audio file after transcription
            if os.path.exists(video_path):
                os.remove(video_path)

            if os.path.exists("videos"):
                shutil.rmtree("videos")

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
