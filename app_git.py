# ===============================================
# 🎙️ Mongolian Fast-Whisper STT (v2.3 — Cloud-Stable Mobile Fast, Guarded)
# ✅ WebM/Opus via PyAV (no ffmpeg), single-read I/O, safe inference & retry
# ===============================================

import streamlit as st
from faster_whisper import WhisperModel
import soundfile as sf
import numpy as np
import scipy.signal
import tempfile
import platform
import time
import io
import traceback
import av  # ✅ PyAV decoder

# ===============================================
# --- PAGE SETUP & STYLING ---
# ===============================================
st.set_page_config(
    page_title="🎙️ Mongolian Fast-Whisper STT",
    page_icon="🎧",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+MN&display=swap');
html,body,[class*="css"]{
    font-family:'Noto Sans MN',sans-serif;
    background:radial-gradient(circle at top left,#f0f2f6,#dfe4ea);
    color:#222;
}
h1{
    background:-webkit-linear-gradient(45deg,#0f4c81,#1f8ac0);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    font-weight:800;
    text-align:center;
    margin-bottom:0.3rem;
}
.subtitle{text-align:center;font-size:1.1rem;color:#555;margin-bottom:1rem;font-style:italic;}
div.stButton>button:first-child{
    background:linear-gradient(90deg,#0f4c81,#1f8ac0);
    color:white;font-weight:bold;border-radius:12px;padding:0.6rem 1.2rem;border:none;
}
.stSuccess,.stInfo,.stWarning,.stError{border-radius:10px;}
[data-testid="stAudioInput"] {
    margin-top: 1.2rem;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: center;
}
button[data-testid="stAudioInput__record"] {
    transform: scale(1.5);
    background: linear-gradient(90deg,#0f4c81,#1f8ac0);
    color: white !important;
    border-radius: 50%;
    border: none;
    width: 80px !important;
    height: 80px !important;
    font-size: 1.1rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
}
button[data-testid="stAudioInput__stop"] {
    transform: scale(1.5);
    background: linear-gradient(90deg,#d32f2f,#f44336);
    color: white !important;
    border-radius: 50%;
    border: none;
    width: 80px !important;
    height: 80px !important;
    font-size: 1.1rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1>🎙️ Mongolian Fast-Whisper STT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>(Anti-Hallucination Edition — Cloud-Stable)</p>", unsafe_allow_html=True)
st.caption("⚡ Fine-tuned Mongolian Whisper model with stable cloud inference")

# ===============================================
# --- MODEL LOADING (CACHED) ---
# ===============================================
system = platform.system().lower()
processor = platform.processor().lower()
compute_type = "float32" if ("darwin" in system and "apple" in processor) else "int8"

@st.cache_resource(show_spinner=False)
def load_model():
    repo_id = "gana1215/MN_Whisper_Small_CT2"
    model = WhisperModel(repo_id, device="cpu", compute_type=compute_type)
    return model

with st.spinner("🔁 Loading Whisper model..."):
    model = load_model()
st.success("✅ Model loaded successfully! Ready to transcribe your voice.")

# ===============================================
# --- HELPERS ---
# ===============================================
def decode_webm_to_float32_mono_16k(webm_bytes: bytes):
    """Decode WebM/Opus to float32 mono @ 16k using PyAV (no external ffmpeg)."""
    container = av.open(io.BytesIO(webm_bytes))
    astream = next(s for s in container.streams if s.type == "audio")
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    for packet in container.demux(astream):
        for frame in packet.decode():
            f = resampler.resample(frame)
            arr = f.to_ndarray()
            if arr.ndim > 1:
                arr = np.mean(arr, axis=0)
            chunks.append(arr.astype(np.float32) / 32768.0)
    container.close()
    if not chunks:
        return np.zeros((0,), dtype=np.float32), 16000
    return np.concatenate(chunks), 16000

def ensure_mono_16k(data: np.ndarray, sr: int):
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != 16000:
        data = scipy.signal.resample_poly(data, 16000, sr)
        sr = 16000
    return np.nan_to_num(data.astype(np.float32)), sr

# ===============================================
# --- AUDIO RECORDING SECTION ---
# ===============================================
st.subheader("🎤 Record your voice below")
st.write("Click the mic icon, speak in Mongolian, then click stop to transcribe:")

audio_file = st.audio_input("🎙️ Start recording")

if audio_file is not None:
    st.success(f"🎧 Recorded audio received — {audio_file.size} bytes")
    st.caption(f"📁 MIME type: {audio_file.type}")

    try:
        audio_bytes = audio_file.read()
        time.sleep(0.3)  # allow full buffer flush

        if not audio_bytes or len(audio_bytes) < 1000:
            st.warning("⚠️ Empty or too short recording — please try again.")
            st.stop()

        # --- Decode audio ---
        try:
            if audio_file.type and "webm" in audio_file.type:
                st.info("🔄 Converting WebM/Opus → 16 kHz mono (PyAV)…")
                data, sr = decode_webm_to_float32_mono_16k(audio_bytes)
            else:
                data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                data, sr = ensure_mono_16k(data, sr)
        except Exception as e:
            st.error(f"❌ Audio decode failed: {e}")
            st.stop()

        if len(data) < sr * 0.3:
            st.warning("⚠️ Recording too short (<0.3 s). Please speak a bit longer.")
            st.stop()

        st.caption(f"📊 Decoded: shape={data.shape}, sr={sr} Hz")

        # --- Save temp WAV ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp, data, sr)
            tmp_path = tmp.name

        # --- Prevent concurrent inference ---
        if st.session_state.get("is_transcribing", False):
            st.warning("⚙️ Already transcribing… please wait.")
            st.stop()
        st.session_state["is_transcribing"] = True

        # --- Transcribe safely ---
        st.info("⏳ Recognizing your Mongolian speech…")
        t0 = time.time()
        try:
            segments, info = model.transcribe(
                tmp_path,
                language="mn",
                beam_size=1,
                vad_filter=True,
                suppress_tokens="-1",
                condition_on_previous_text=False,
                word_timestamps=False,
                max_batch_size=1,
                num_workers=1
            )
        except Exception as e:
            st.error("❌ Transcription failed.")
            st.exception(e)
            traceback.print_exc()
            st.session_state["is_transcribing"] = False
            st.stop()

        # --- Collect result ---
        text = " ".join([s.text.strip() for s in segments if s.text.strip()]) if segments else ""
        dt = time.time() - t0
        st.session_state["is_transcribing"] = False

        # --- Output ---
        if text:
            st.success("✅ Recognition complete!")
            st.markdown("### 🗣️ Recognized Text:")
            st.markdown(
                f"<div style='padding:1rem;background:#f8f9fa;border-radius:12px;"
                f"font-size:1.3rem;color:#111;'>{text}</div>",
                unsafe_allow_html=True
            )
            st.caption(f"⚡ {dt:.2f}s — Model: MN_Whisper_Small_CT2 ({compute_type})")
        else:
            st.warning("⚠️ No speech detected. Please try again closer to the mic.")

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.session_state["is_transcribing"] = False
        traceback.print_exc()

else:
    st.info("⏺️ Waiting for you to record…")

# ===============================================
# --- FOOTER ---
# ===============================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#666;'>Developed by <b>Gankhuyag Mambaryenchin</b><br>"
    "Fine-tuned Whisper Model — Mongolian Fast-Whisper (Anti-Hallucination Edition v2.3 Guarded)</p>",
    unsafe_allow_html=True
)
