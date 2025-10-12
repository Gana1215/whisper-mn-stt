# ===============================================
# 🎙️ Whisper MN Streamlit App (Cloud + Local)
# ===============================================

import os
import io
import zipfile
import requests
import torch
import streamlit as st
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration

# --- Constants ---
MODEL_DIR = "./models/checkpoint-3500"
MODEL_ZIP_URL = "https://www.dropbox.com/scl/fi/k33mfgw2r05we2t636zgi/checkpoint-3500.zip?rlkey=s5x0os8hoktpu1mbzbg5pfhoy&st=zxpg2fxn&dl=1"
os.makedirs("./models", exist_ok=True)

# ======================================================
# 🔽 Download + Extract Model (only once per fresh start)
# ======================================================
def download_and_extract_model():
    if os.path.exists(MODEL_DIR):
        st.success(f"✅ Model already exists at {MODEL_DIR}")
        return

    zip_path = "./models/checkpoint-3500.zip"
    st.info("⬇️ Downloading model from Dropbox...")

    response = requests.get(MODEL_ZIP_URL, stream=True)
    if response.status_code != 200:
        st.error(f"❌ Download failed (status {response.status_code})")
        return

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1MB
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(block_size):
            if chunk:
                f.write(chunk)
    st.success("✅ Download complete!")

    st.info("📦 Extracting model files...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("./models")

    if os.path.exists(MODEL_DIR):
        st.success("✅ Model extracted successfully!")
    else:
        st.error("❌ Extraction failed — MODEL_DIR not found after unzip.")

# ======================================================
# 🧭 Debug Info (see working dir + files in Streamlit Cloud)
# ======================================================
st.markdown("### 🧭 Environment Debug Info")
st.write(f"**Current Working Directory:** `{os.getcwd()}`")

dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
files = [f for f in os.listdir('.') if os.path.isfile(f)]
st.write("📂 **Folders in root:**", dirs)
st.write("📄 **Files in root:**", files)

# ======================================================
# ⚙️ Ensure model is available
# ======================================================
download_and_extract_model()

if not os.path.exists(MODEL_DIR):
    st.error(f"❌ MODEL_DIR not found: {MODEL_DIR}")
    st.write("📂 Available inside ./models:", os.listdir("./models") if os.path.exists("./models") else "models folder missing")
    st.stop()
else:
    files = os.listdir(MODEL_DIR)
    st.success(f"✅ MODEL_DIR found: {MODEL_DIR}")
    st.write(f"📦 Found {len(files)} files:", files[:10])

# ======================================================
# 🧠 Load Whisper Model (with fallback)
# ======================================================
try:
    processor = WhisperProcessor.from_pretrained(MODEL_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
    st.success("✅ Fine-tuned Whisper model loaded successfully!")
except Exception as e:
    st.warning(f"⚠️ Could not load fine-tuned model: {e}")
    st.info("Loading fallback model: openai/whisper-tiny ...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# ======================================================
# 🎙️ App Interface
# ======================================================
st.title("🎧 Mongolian Whisper Speech-to-Text")
st.write("Record or upload Mongolian audio and transcribe using Whisper!")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    # Save temp file
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Load + transcribe
    try:
        pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer)
        st.info("⏳ Transcribing...")
        result = pipe("temp.wav", generate_kwargs={"task": "transcribe", "language": "mn"})
        st.success("✅ Transcription complete:")
        st.text_area("Recognized text:", result["text"], height=200)
    except Exception as e:
        st.error(f"❌ Error during transcription: {e}")
