# ===============================================
# 🎧 Mongolian Whisper Speech-to-Text (Cloud Edition)
# ===============================================
import os
import time
import torch
import tempfile
import zipfile
import requests
import soundfile as sf
import streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ------------------- Streamlit Setup -------------------
st.set_page_config(page_title="Mongolian Whisper STT", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+MN&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans MN', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("🎧 Mongolian Whisper Speech-to-Text")
st.write("Record or upload Mongolian audio and transcribe using your fine-tuned Whisper model!")

# ===============================================
# 🧭 Environment Info
# ===============================================
cwd = os.getcwd()
st.write("🧭 Environment Debug Info")
st.write(f"Current Working Directory: {cwd}")

st.write("📂 Folders in root:")
st.write(os.listdir("."))

# ===============================================
# 📦 Model Download + Extraction
# ===============================================
MODEL_ZIP_URL = "https://www.dropbox.com/scl/fi/8nmh0twbvhjvrxdvyui0t/checkpoint-3500.zip?rlkey=klfvnm6dble9oxsplwa03y42h&st=xpmhlgr0&dl=1"
#https://www.dropbox.com/scl/fi/8nmh0twbvhjvrxdvyui0t/checkpoint-3500.zip?rlkey=klfvnm6dble9oxsplwa03y42h&st=xpmhlgr0&dl=1
BASE_MODEL_DIR = "./models"
MODEL_DIR = os.path.join(BASE_MODEL_DIR, "checkpoint-3500")
MODEL_ZIP = os.path.join(BASE_MODEL_DIR, "checkpoint-3500.zip")

os.makedirs(BASE_MODEL_DIR, exist_ok=True)

def download_and_extract_model():
    """Download and extract model zip from Dropbox if needed."""
    if not os.path.exists(MODEL_ZIP):
        st.write("⬇️ Downloading fine-tuned model from Dropbox...")
        r = requests.get(MODEL_ZIP_URL, stream=True)
        with open(MODEL_ZIP, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success("✅ Download complete!")

    if not os.path.exists(MODEL_DIR):
        st.write("📦 Extracting model files...")
        with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
            zip_ref.extractall(BASE_MODEL_DIR)
        st.success("✅ Extraction complete!")
    else:
        st.success("✅ Model already extracted")

download_and_extract_model()

# Debug
if os.path.exists(MODEL_DIR):
    files = os.listdir(MODEL_DIR)
    st.write(f"✅ MODEL_DIR found: {MODEL_DIR}")
    st.write(f"📦 Found {len(files)} files:")
    st.write(files)
else:
    st.error(f"❌ MODEL_DIR not found: {MODEL_DIR}")

# ===============================================
# 🤖 Load Whisper Model
# ===============================================
try:
    processor = WhisperProcessor.from_pretrained(MODEL_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
    st.success("✅ Fine-tuned model loaded successfully!")
except Exception as e:
    st.warning(f"⚠️ Could not load fine-tuned model: {e}")
    st.info("Loading fallback model: openai/whisper-tiny ...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# Anti-hallucination decoding setup
forced_ids = processor.get_decoder_prompt_ids(language="mn", task="transcribe")
generate_kwargs = {
    "forced_decoder_ids": forced_ids,
    "do_sample": False,
    "temperature": 0.0,
    "num_beams": 5,
    "max_new_tokens": 64,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.05,
    "length_penalty": 0.1,
}

# ===============================================
# 🧹 Silence Trimmer
# ===============================================
def trim_silence(audio_tensor, sr=16000, thresh=0.005):
    frame_len = int(sr * 0.03)
    frames = audio_tensor.unfold(0, frame_len, frame_len)
    energy = (frames ** 2).mean(dim=1)
    mask = energy > thresh
    if not mask.any():
        return audio_tensor
    mask_int = mask.int()
    start = (mask_int.argmax().item()) * frame_len
    end = (len(mask_int) - torch.flip(mask_int, [0]).argmax().item()) * frame_len
    return audio_tensor[start:end].contiguous()

# ===============================================
# 🎤 Upload + Transcribe
# ===============================================
st.markdown("---")
st.subheader("🎙️ Upload a .wav file")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Load and process audio
    try:
        data, sr = sf.read(temp_path)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        data = torch.tensor(data, dtype=torch.float32)
        data = trim_silence(data)
        sr = 16000
        sf.write(temp_path, data.numpy(), sr)
    except Exception as e:
        st.error(f"❌ Failed to read audio file: {e}")
        st.stop()

    # Perform inference directly on waveform
    st.info("⏳ Recognizing your voice... please wait")
    try:
        inputs = processor(data.numpy(), sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            predicted_ids = model.generate(**inputs, **generate_kwargs)
        result_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        st.success("✅ Recognition complete!")
        st.markdown("### 🗣️ Recognized Text:")
        st.markdown(f"<h2>{result_text}</h2>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error during transcription: {e}")
