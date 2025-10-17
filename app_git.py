# ===============================================
# 🎙️ Mongolian Fast-Whisper STT (Anti-Hallucination Edition)
# Version 1.8 — Cloud-Stable (Hugging Face Integrated)
# ===============================================
import streamlit as st
from st_audiorec import st_audiorec
from transformers import WhisperProcessor
import ctranslate2
import soundfile as sf
import numpy as np
import tempfile, time, platform

# --- Page Setup ---
st.set_page_config(page_title="🎙️ Mongolian Fast-Whisper STT", page_icon="🎧", layout="centered")

# --- Elegant Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+MN&display=swap');
html,body,[class*="css"]{font-family:'Noto Sans MN',sans-serif;
background:radial-gradient(circle at top left,#f0f2f6,#dfe4ea);color:#222;}
h1{background:-webkit-linear-gradient(45deg,#0f4c81,#1f8ac0);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
font-weight:800;text-align:center;margin-bottom:0.3rem;}
.subtitle{text-align:center;font-size:1.1rem;color:#555;margin-bottom:1rem;font-style:italic;}
div.stButton>button:first-child{background:linear-gradient(90deg,#0f4c81,#1f8ac0);
color:white;font-weight:bold;border-radius:12px;padding:0.6rem 1.2rem;border:none;}
.stSuccess,.stInfo,.stWarning,.stError{border-radius:10px;}
div[data-testid="stAudio"]{display:none;}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1>🎙️ Mongolian Fast-Whisper STT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>(Whisper Anti-Hallucination Edition)</p>", unsafe_allow_html=True)
st.caption("⚡ Fast, accurate, and locally fine-tuned for Mongolian speech recognition")
st.markdown("<small style='color:gray;'>🎤 If the microphone doesn’t start, refresh and allow mic access.</small>", unsafe_allow_html=True)

# --- Dynamic compute type ---
system = platform.system().lower()
processor = platform.processor().lower()
if "darwin" in system and "apple" in processor: compute_type = "float32"
elif "darwin" in system: compute_type = "int8"
else: compute_type = "int8"  # Cloud/Colab CPU

# --- Cached model loader ---
#@st.cache_resource(show_spinner=False)

@st.cache_resource(show_spinner=False)
def load_model():
    repo_id = "gana1215/MN_Whisper_Small_CT2"
    processor = WhisperProcessor.from_pretrained(repo_id)
    translator = ctranslate2.Translator(repo_id, device="cpu", compute_type=compute_type)
    return processor, translator

with st.spinner("🔁 Loading Fast-Whisper model..."):
    processor, translator = load_model()
st.success("✅ Model loaded successfully! Ready to transcribe your voice.")

# --- Audio Recorder ---
st.subheader("🎤 Record your voice below")
st.write("Click the red circle, speak in Mongolian, then click Stop to transcribe:")
wav_audio_data = st_audiorec()
st.write("🎧 Audio bytes length:", len(wav_audio_data) if wav_audio_data else 0)

# --- Transcription ---
def transcribe(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    audio, sr = sf.read(tmp_path)
    features = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="np").input_features
    tokens = translator.translate_batch([processor.tokenizer.bos_token_id], max_decoding_length=448)
    decoded = processor.tokenizer.decode(tokens[0]["tokens"], skip_special_tokens=True)
    return decoded.strip()

if wav_audio_data:
    st.markdown("---")
    st.info("⏳ Recognizing your speech... please wait")
    start = time.time()
    recognized_text = transcribe(wav_audio_data)
    end = time.time()
    if recognized_text:
        st.success("✅ Recognition complete!")
        st.markdown("### 🗣️ Recognized Text:")
        st.markdown(f"<div style='padding:1rem;background:#f8f9fa;border-radius:12px;"
                    f"font-size:1.3rem;color:#111;'>{recognized_text}</div>", unsafe_allow_html=True)
        st.caption(f"⚡ {end - start:.2f}s processing — Model: MN_Whisper_Small_CT2 ({compute_type})")
    else:
        st.warning("⚠️ No speech detected. Try speaking louder or closer to the mic.")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align:center;color:#666;'>Developed by <b>Gankhuyag Mambaryenchin</b><br>"
            "Fine-tuned Whisper Model — Mongolian Fast-Whisper (Anti-Hallucination Edition)</p>",
            unsafe_allow_html=True)
