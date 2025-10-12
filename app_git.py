# ===============================================
# 🎧 Mongolian Whisper Speech-to-Text (Cloud Edition)
# ===============================================
import os
import torch
import tempfile
import zipfile
import requests
import soundfile as sf
import streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# --- Optional recorder ---
try:
    from st_audiorec import st_audiorec
    HAS_RECORDER = True
except ImportError:
    HAS_RECORDER = False

# ===============================================
# 🌐 Streamlit Setup
# ===============================================
st.set_page_config(page_title="Mongolian Whisper STT", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+MN&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans MN', sans-serif; }
h1, h2, h3 { color: #1a73e8; text-align: center; }
div[data-testid="stMarkdownContainer"] h2 {
    background-color: #f1f3f4; border-radius: 12px; padding: 10px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🎧 Mongolian Whisper Speech-to-Text")
st.caption("Fine-tuned Whisper model for Mongolian audio recognition")

# ===============================================
# 📦 Model Download + Extraction
# ===============================================
MODEL_ZIP_URL = "https://www.dropbox.com/scl/fi/8nmh0twbvhjvrxdvyui0t/checkpoint-3500.zip?rlkey=klfvnm6dble9oxsplwa03y42h&st=xpmhlgr0&dl=1"
BASE_MODEL_DIR = "./models"
MODEL_DIR = os.path.join(BASE_MODEL_DIR, "checkpoint-3500")
MODEL_ZIP = os.path.join(BASE_MODEL_DIR, "checkpoint-3500.zip")
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

def download_and_extract_model():
    if not os.path.exists(MODEL_ZIP):
        st.info("⬇️ Downloading fine-tuned model from Dropbox...")
        r = requests.get(MODEL_ZIP_URL, stream=True)
        with open(MODEL_ZIP, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk: f.write(chunk)
        st.success("✅ Download complete!")

    if not os.path.exists(MODEL_DIR):
        st.info("📦 Extracting model files...")
        with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
            zip_ref.extractall(BASE_MODEL_DIR)
        st.success("✅ Extraction complete!")
    else:
        st.success("✅ Model ready")

download_and_extract_model()

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
# 🧹 Silence Trimmer (safe)
# ===============================================
def trim_silence(audio_tensor, sr=16000, thresh=0.005):
    """Trim leading and trailing silence safely (handles short clips)."""
    if not isinstance(audio_tensor, torch.Tensor):
        audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)
    if audio_tensor.numel() == 0:
        return audio_tensor
    # normalize to avoid thresh mis-detection
    maxv = audio_tensor.abs().max()
    if maxv > 0:
        audio_tensor = audio_tensor / (maxv + 1e-8)

    frame_len = int(sr * 0.03)  # 30ms
    if audio_tensor.numel() < frame_len:
        return audio_tensor  # too short

    try:
        frames = audio_tensor.unfold(0, frame_len, frame_len)
    except RuntimeError:
        return audio_tensor

    energy = (frames ** 2).mean(dim=1)
    mask = energy > thresh
    if not mask.any():
        return audio_tensor

    start = (mask.int().argmax().item()) * frame_len
    end = (len(mask) - torch.flip(mask.int(), [0]).argmax().item()) * frame_len
    trimmed = audio_tensor[start:end].contiguous()
    return trimmed if trimmed.numel() > 0 else audio_tensor

# ===============================================
# 🧠 Session State
# ===============================================
def reset_states():
    st.session_state["recognized_text_record"] = ""
    st.session_state["recognized_text_upload"] = ""
    st.session_state["record_audio_bytes"] = None
    st.session_state["is_processing"] = False

# init
for k, v in {
    "recognized_text_record": "",
    "recognized_text_upload": "",
    "record_audio_bytes": None,
    "is_processing": False,
    "mode": "🎙️ Record Voice",
}.items():
    st.session_state.setdefault(k, v)

def on_mode_change():
    reset_states()

# ===============================================
# 🎛 Mode Selector (no rerun, clean reset)
# ===============================================
st.markdown("---")
mode = st.selectbox(
    "Select Mode",
    ["🎙️ Record Voice", "📂 Upload Audio File"],
    index=0 if st.session_state["mode"] == "🎙️ Record Voice" else 1,
    key="mode",
    on_change=on_mode_change,
)

# ===============================================
# 🧩 helpers: run transcription
# ===============================================
def transcribe_waveform(waveform, sr=16000):
    status = st.info("⏳ Converting to text...")
    try:
        inputs = processor(waveform.numpy(), sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            predicted_ids = model.generate(**inputs, **generate_kwargs)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        status.empty()
        st.success("✅ Recognition complete!")
        return text
    except Exception as e:
        status.empty()
        st.error(f"❌ Error during transcription: {e}")
        return ""

def load_and_prepare_wav(path):
    data, sr = sf.read(path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    data = torch.tensor(data, dtype=torch.float32)
    data = trim_silence(data, sr=sr)
    # resample to 16k? soundfile doesn't resample; keep sr as-is if model supports 16k frontend.
    # Whisper expects 16k features; processor will handle resampling internally if needed in latest versions,
    # but to be safe we write back at 16k.
    target_sr = 16000
    sf.write(path, data.numpy(), target_sr)
    return data, target_sr

# ===============================================
# 🎤 RECORD MODE (explicit button to transcribe after STOP)
# ===============================================
if mode == "🎙️ Record Voice":
    if not HAS_RECORDER:
        st.error("⚠️ Recorder not available in this environment.")
    else:
        st.subheader("🎙️ Record your voice below:")
        wav_audio_data = st_audiorec()

        # store the last completed recording bytes (component returns bytes after STOP)
        if wav_audio_data and len(wav_audio_data) > 0:
            st.session_state["record_audio_bytes"] = wav_audio_data

        # show preview & transcribe button only when we have a completed recording
        if st.session_state["record_audio_bytes"] is not None:
            st.audio(st.session_state["record_audio_bytes"], format="audio/wav")
            col1, col2 = st.columns([1,1])
            with col1:
                do_transcribe = st.button("📝 Transcribe recording", disabled=st.session_state["is_processing"])
            with col2:
                if st.button("🔁 Reset recording"):
                    st.session_state["record_audio_bytes"] = None
                    st.session_state["recognized_text_record"] = ""

            if do_transcribe:
                st.session_state["is_processing"] = True
                # write to temp and process
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(st.session_state["record_audio_bytes"])
                    temp_path = tmp.name
                try:
                    data, sr = load_and_prepare_wav(temp_path)
                    text = transcribe_waveform(data, sr)
                    st.session_state["recognized_text_record"] = text
                finally:
                    st.session_state["is_processing"] = False

        # show recognized text (editable)
        if st.session_state["recognized_text_record"]:
            st.markdown("### 🗣️ Recognized Text:")
            final_text = st.text_area(
                "📝 Edit or copy recognized text:",
                st.session_state["recognized_text_record"],
                height=150,
                key="record_text_area"
            )
            st.download_button("💾 Save text result", final_text, "recognized_text_record.txt", mime="text/plain")

# ===============================================
# 📂 UPLOAD MODE (explicit button to transcribe)
# ===============================================
elif mode == "📂 Upload Audio File":
    st.subheader("📂 Upload a .wav file for transcription:")
    uploaded_file = st.file_uploader("Upload your audio file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        do_transcribe = st.button("📝 Transcribe file", disabled=st.session_state["is_processing"])
        if do_transcribe:
            st.session_state["is_processing"] = True
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name
            try:
                data, sr = load_and_prepare_wav(temp_path)
                text = transcribe_waveform(data, sr)
                st.session_state["recognized_text_upload"] = text
            finally:
                st.session_state["is_processing"] = False

    if st.session_state["recognized_text_upload"]:
        st.markdown("### 🗣️ Recognized Text:")
        final_text = st.text_area(
            "📝 Edit or copy recognized text:",
            st.session_state["recognized_text_upload"],
            height=150,
            key="upload_text_area"
        )
        st.download_button("💾 Save text result", final_text, "recognized_text_upload.txt", mime="text/plain")
