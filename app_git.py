# ===============================================
# 🎙️ Whisper Mongolian STT — Cloud Safe Version
# ===============================================
import os
import torch
import torchaudio
import tempfile
import streamlit as st
import requests
import zipfile
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration

# --- Streamlit page setup ---
st.set_page_config(page_title="Whisper Mongolian STT", layout="centered")
st.title("🎙️ Mongolian Speech-to-Text (Whisper Anti-Hallucination Edition)")

# --- Dropbox public link ---
DROPBOX_MODEL_URL = "https://www.dropbox.com/scl/fo/sruai8kxjto7b9qaq334f/AOQ7s1VL6nGLVjgGS3VSLOM?rlkey=52dakrqa4zfbqknhibv0tcrlt&dl=1"

MODEL_DIR = "./models/checkpoint-3500"
# ===============================================
# 📦 Model Downloader + Safe Loader (Dropbox → Local)
# ===============================================
import os, zipfile, requests, streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_DIR = "./models/checkpoint-3500"
MODEL_ZIP = "./checkpoint-3500.zip"

# 🔗 Dropbox direct link (note: use ?dl=1 at the end!)
DROPBOX_URL = "https://www.dropbox.com/scl/fo/sruai8kxjto7b9qaq334f/AOQ7s1VL6nGLVjgGS3VSLOM?rlkey=52dakrqa4zfbqknhibv0tcrlt&dl=1"

def download_and_extract_model():
    """Download fine-tuned Whisper model from Dropbox if not present locally."""
    tokenizer_file = os.path.join(MODEL_DIR, "tokenizer.json")

    if os.path.exists(tokenizer_file):
        st.info("✅ Model already exists locally — skipping download.")
        return

    st.info("📥 Downloading model from Dropbox (first-time only)...")
    os.makedirs("models", exist_ok=True)

    # Download with progress
    r = requests.get(DROPBOX_URL, stream=True)
    total = int(r.headers.get("content-length", 0))
    with open(MODEL_ZIP, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

    # Extract zip
    with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
        zip_ref.extractall("models")
    os.remove(MODEL_ZIP)
    st.success("✅ Model downloaded and extracted!")

# --- Run setup ---
download_and_extract_model()

# --- Load Whisper model safely ---
try:
    processor = WhisperProcessor.from_pretrained(MODEL_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
    st.success("✅ Fine-tuned Whisper model loaded!")
except Exception as e:
    st.error(f"⚠️ Could not load fine-tuned model: {e}")
    st.warning("Loading fallback model: openai/whisper-tiny")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# --- Load Whisper model ---
processor = WhisperProcessor.from_pretrained(MODEL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)

forced_ids = processor.get_decoder_prompt_ids(language="mn", task="transcribe")
generate_kwargs = {
    "forced_decoder_ids": forced_ids,
    "do_sample": False,
    "temperature": 0.0,
    "num_beams": 5,
    "max_new_tokens": 32,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.05,
    "length_penalty": 0.1,
}
asr = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=15,
    stride_length_s=(2, 2),
    generate_kwargs=generate_kwargs,
    device=0 if torch.cuda.is_available() else -1,
)
st.success("✅ Model loaded and ready!")

# --- File uploader ---
uploaded_file = st.file_uploader("🎤 Upload your voice file (WAV/MP3)", type=["wav", "mp3"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Load + preprocess
    wav, sr = torchaudio.load(temp_path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    wav = torchaudio.functional.resample(wav, sr, 16000)
    torchaudio.save(temp_path, wav, 16000)

    st.info("⏳ Recognizing speech...")
    result = asr(temp_path)
    st.success("✅ Recognition complete!")
    st.markdown(f"### 🗣️ Recognized Text:\n\n**{result['text']}**")
