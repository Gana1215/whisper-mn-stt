# ===============================================
# 🎧 Mongolian Whisper Speech-to-Text (Auto-Transcribe Fixed)
# ===============================================
import os
import torch
import tempfile
import zipfile
import requests
import soundfile as sf
import numpy as np
import streamlit as st
from transformers import AutoProcessor, WhisperForConditionalGeneration

try:
    from st_audiorec import st_audiorec
    HAS_RECORDER = True
except ImportError:
    HAS_RECORDER = False

# ===============================================
# 🌐 Streamlit Setup
# ===============================================
st.set_page_config(page_title="Mongolian Whisper STT", layout="centered")
st.title("🎧 Mongolian Whisper Speech-to-Text")
st.caption("Fine-tuned Whisper model for Mongolian audio recognition")

# ===============================================
# 📦 Model Download + Extraction
# ===============================================
MODEL_ZIP_URL = "https://www.dropbox.com/scl/fi/8nmh0twbvhjvrxdvyui0t/checkpoint-3500.zip?rlkey=klfvnm6dble9oxsplwa03y42h&dl=1"
BASE_MODEL_DIR = "./models"
MODEL_DIR = os.path.join(BASE_MODEL_DIR, "checkpoint-3500")
MODEL_ZIP = os.path.join(BASE_MODEL_DIR, "checkpoint-3500.zip")
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

def download_and_extract_model():
    if not os.path.exists(MODEL_ZIP):
        st.info("⬇️ Downloading model...")
        r = requests.get(MODEL_ZIP_URL, stream=True)
        with open(MODEL_ZIP, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        st.success("✅ Download complete!")
    if not os.path.exists(MODEL_DIR):
        with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
            zip_ref.extractall(BASE_MODEL_DIR)
        st.success("✅ Extracted fine-tuned model!")

download_and_extract_model()

# ===============================================
# 🤖 Load Whisper
# ===============================================
try:
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
    st.success("✅ Fine-tuned model loaded successfully!")
except Exception as e:
    st.warning(f"⚠️ Model load issue: {e}")
    st.info("Fallback: openai/whisper-tiny")
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

forced_ids = processor.get_decoder_prompt_ids(language="mn", task="transcribe")

# ===============================================
# 🔊 Utility: Safe trim
# ===============================================
def trim_silence(audio_tensor, sr=16000, thresh=0.005):
    if not isinstance(audio_tensor, torch.Tensor):
        audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)
    frame_len = int(sr * 0.03)
    if len(audio_tensor) < frame_len:
        return audio_tensor
    frames = audio_tensor.unfold(0, frame_len, frame_len)
    energy = (frames ** 2).mean(dim=1)
    mask = energy > thresh
    if not mask.any():
        return audio_tensor
    start = (mask.int().argmax().item()) * frame_len
    end = (len(mask) - torch.flip(mask.int(), [0]).argmax().item()) * frame_len
    return audio_tensor[start:end].contiguous()

# ===============================================
# 🔠 Transcribe Helper
# ===============================================
def transcribe_audio_array(audio_data: np.ndarray, sr: int = 16000):
    st.info("⏳ Converting to text...")
    try:
        inputs = processor(audio_data, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            pred = model.generate(**inputs, forced_decoder_ids=forced_ids)
        text = processor.batch_decode(pred, skip_special_tokens=True)[0]
        st.success("✅ Transcription complete!")
        return text.strip()
    except Exception as e:
        st.error(f"❌ Transcription failed: {e}")
        return ""

# ===============================================
# 🎙️ Record Voice Mode
# ===============================================
st.markdown("---")
mode = st.radio("Select Mode", ["🎙️ Record Voice", "📂 Upload Audio"], horizontal=True)

if mode == "🎙️ Record Voice":
    if not HAS_RECORDER:
        st.error("⚠️ Recorder not supported here.")
    else:
        st.subheader("🎙️ Record and speak...")
        wav_audio_data = st_audiorec()

        if wav_audio_data:
            st.info("📥 Recording finished, processing audio...")

            # 🔧 Convert raw bytes → valid WAV
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(wav_audio_data)
                wav_path = tmp.name

            try:
                data, sr = sf.read(wav_path)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                st.write(f"🎧 Audio loaded: {len(data)/sr:.2f} seconds, {sr} Hz")

                # Trim & resample if needed
                data = trim_silence(torch.tensor(data))
                sf.write(wav_path, data.numpy(), 16000)

                st.audio(wav_audio_data, format="audio/wav")

                # 🧠 Transcribe automatically after STOP
                text = transcribe_audio_array(data.numpy(), 16000)
                if text:
                    st.markdown("### 🗣️ Recognized Text:")
                    st.text_area("Edit recognized text:", text, height=150)
                    st.download_button("💾 Save result", text, "record_transcription.txt")

            except Exception as e:
                st.error(f"❌ Failed to read/process recording: {e}")

# ===============================================
# 📂 Upload Mode
# ===============================================
elif mode == "📂 Upload Audio":
    st.subheader("📂 Upload a WAV file:")
    uploaded = st.file_uploader("Upload .wav", type=["wav"])

    if uploaded:
        st.audio(uploaded, format="audio/wav")
        if st.button("📝 Transcribe File"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded.read())
                path = tmp.name
            data, sr = sf.read(path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = trim_silence(torch.tensor(data))
            text = transcribe_audio_array(data.numpy(), sr)
            if text:
                st.text_area("🗣️ Recognized Text:", text, height=150)
                st.download_button("💾 Save", text, "upload_transcription.txt")
