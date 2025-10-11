# ===============================================
# 🎙️ Whisper Mongolian STT — Streamlit Cloud + Dropbox
# ===============================================
import os
import io
import torch
import torchaudio
import tempfile
import streamlit as st
import dropbox
from st_audiorec import st_audiorec
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration

# --- Streamlit Config ---
st.set_page_config(page_title="Whisper Mongolian STT", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+MN&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans MN', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("🎙️ Mongolian Speech-to-Text (Whisper Anti-Hallucination Edition)")

# ===============================================
# 🔐 Dropbox Setup
# ===============================================
DROPBOX_TOKEN = st.secrets["DROPBOX_TOKEN"]
MODEL_FOLDER = "/models/checkpoint-3500"
LOCAL_MODEL_DIR = "./models/checkpoint-3500"

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

@st.cache_resource(show_spinner=True)
def download_model_from_dropbox():
    """Download Whisper model files from Dropbox (only if not cached)."""
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)
    st.info("🔄 Checking and downloading model from Dropbox...")

    for entry in dbx.files_list_folder(MODEL_FOLDER).entries:
        if isinstance(entry, dropbox.files.FileMetadata):
            file_name = entry.name
            local_path = os.path.join(LOCAL_MODEL_DIR, file_name)
            if not os.path.exists(local_path):
                _, res = dbx.files_download(path=entry.path_display)
                with open(local_path, "wb") as f:
                    f.write(res.content)
    return LOCAL_MODEL_DIR

MODEL_PATH = download_model_from_dropbox()
st.success("✅ Model downloaded successfully!")

# ===============================================
# 🎧 Load Whisper Model
# ===============================================
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

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
st.success("✅ Whisper model loaded and ready!")

# ===============================================
# 🎚️ Silence Trimming Helper
# ===============================================
def trim_silence(audio, sr=16000, thresh=0.005):
    frame_len = int(sr * 0.03)
    frames = audio.unfold(0, frame_len, frame_len)
    energy = (frames ** 2).mean(dim=1)
    mask = energy > thresh

    if not mask.any():
        return audio
    mask_int = mask.int()
    start = (mask_int.argmax().item()) * frame_len
    end = (len(mask_int) - torch.flip(mask_int, [0]).argmax().item()) * frame_len
    return audio[start:end].contiguous()

# ===============================================
# 🎤 Audio Recorder + Transcription
# ===============================================
st.subheader("🎤 Record your voice below:")
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_audio_data)
        temp_path = tmp.name

    wav, sr = torchaudio.load(temp_path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = trim_silence(wav.squeeze(0))
    torchaudio.save(temp_path, wav.unsqueeze(0), 16000)

    st.markdown("---")
    st.info("⏳ Recognizing your voice... please wait")

    result = asr(temp_path)
    st.success("✅ Recognition complete!")
    st.markdown("### 🗣️ Recognized Text:")
    st.markdown(f"<h2>{result['text']}</h2>", unsafe_allow_html=True)
