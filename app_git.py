# ===============================================
# 🎙️ Mongolian Fast-Whisper STT (v3.3 — Audio-Verified Edition)
# ✅ st_audiorec • Robust decoding • Works on all devices
# ===============================================

import streamlit as st
import torch, io, time, tempfile, platform, concurrent.futures, inspect, os, wave, struct
import numpy as np, soundfile as sf, scipy.signal
from faster_whisper import WhisperModel
from st_audiorec import st_audiorec

# ---------- CPU setup ----------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try: torch.set_num_interop_threads(1)
    except Exception: pass

# ---------- PAGE ----------
st.set_page_config(page_title="🎙️ Mongolian Fast-Whisper STT", page_icon="🎧", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+MN&display=swap');
html,body,[class*="css"]{font-family:'Noto Sans MN',sans-serif;
background:radial-gradient(circle at top left,#f0f2f6,#dfe4ea);color:#222;}
h1{background:-webkit-linear-gradient(45deg,#0f4c81,#1f8ac0);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
font-weight:800;text-align:center;margin-bottom:0.3rem;}
.subtitle{text-align:center;font-size:1.1rem;color:#555;margin-bottom:1rem;font-style:italic;}
div.stButton>button:first-child{
background:linear-gradient(90deg,#0f4c81,#1f8ac0);
color:white;font-weight:bold;border-radius:12px;padding:0.6rem 1.2rem;border:none;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎙️ Mongolian Fast-Whisper STT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>(Anti-Hallucination Edition — Mobile-Stable)</p>", unsafe_allow_html=True)
st.caption("⚡ Fine-tuned Mongolian Whisper model — robust decoding and mobile-ready")

# ---------- MODEL ----------
system, processor = platform.system().lower(), platform.processor().lower()
compute_type = "float32" if ("darwin" in system and "apple" in processor) else "int8"

@st.cache_resource(show_spinner=False)
def load_model():
    return WhisperModel("gana1215/MN_Whisper_Small_CT2", device="cpu", compute_type=compute_type)

with st.spinner("🔁 Loading Whisper model..."):
    model = load_model()
st.success("✅ Model loaded successfully!")

# ---------- HELPERS ----------
def transcribe_compat(model, path, **kwargs):
    sig = inspect.signature(model.transcribe)
    if "show_progress" in sig.parameters: kwargs["show_progress"] = False
    elif "log_progress" in sig.parameters: kwargs["log_progress"] = False
    return model.transcribe(path, **kwargs)

def ensure_mono_16k(data: np.ndarray, sr: int):
    if data.ndim > 1: data = np.mean(data, axis=1)
    if sr != 16000:
        data = scipy.signal.resample_poly(data, 16000, sr)
        sr = 16000
    return data.astype(np.float32), sr

def write_temp_wav(data: np.ndarray, sr: int):
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, data, sr)
    return path

def safe_transcribe(wav_path: str):
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                transcribe_compat, model, wav_path,
                language="mn", beam_size=1,
                vad_filter=True, suppress_tokens=[-1],
                condition_on_previous_text=False,
                word_timestamps=False, temperature=0.0)
            return fut.result(timeout=40)
    except Exception:
        return [], None

# ---------- WAV decoder for st_audiorec ----------
def decode_st_audiorec_bytes(audio_bytes: bytes):
    """Decode the st_audiorec WAV (PCM16) safely to float32 mono @16k."""
    try:
        # Try native decode first
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if len(data) > 100:
            return ensure_mono_16k(data, sr)
    except Exception:
        pass
    # Fallback: manual PCM16 reader
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return ensure_mono_16k(data, sr)

# ---------- WARM-UP ----------
if "warmup_done" not in st.session_state:
    st.session_state["warmup_done"] = True
    try:
        sr = 16000
        warm = np.zeros(int(0.4 * sr), dtype=np.float32)
        path = write_temp_wav(warm, sr)
        transcribe_compat(model, path, language="mn", beam_size=1)
        os.unlink(path)
    except Exception:
        pass

# ---------- STATE ----------
if "last_text" not in st.session_state: st.session_state["last_text"] = ""
if "last_audio_bytes" not in st.session_state: st.session_state["last_audio_bytes"] = None

# ---------- UI ----------
st.subheader("🎤 Record your voice below")
st.write("Press and hold the mic, speak in Mongolian, then release to transcribe:")
audio_bytes = st_audiorec()

# ---------- HANDLER ----------
def handle_audio(audio_bytes: bytes):
    if not audio_bytes or len(audio_bytes) < 800:
        st.warning("🎙️ Recording too short or mic initializing.")
        return
    data, sr = decode_st_audiorec_bytes(audio_bytes)
    if len(data) < sr * 0.3:
        st.warning("⚠️ Recording too short or silent.")
        return
    tmp = write_temp_wav(data, sr)
    st.info("⏳ Recognizing Mongolian speech…")
    t0 = time.time()
    try:
        segments, info = safe_transcribe(tmp)
    finally:
        try: os.unlink(tmp)
        except Exception: pass
    dt = time.time() - t0
    text = " ".join([s.text.strip() for s in segments if getattr(s,'text','').strip()]) if segments else ""
    st.session_state["last_text"] = text
    st.session_state["last_audio_bytes"] = audio_bytes
    if text:
        st.success("✅ Recognition complete!")
        st.markdown(
            f"<div style='padding:1rem;background:#f8f9fa;border-radius:12px;"
            f"font-size:1.3rem;color:#111;'>{text}</div>", unsafe_allow_html=True)
        st.caption(f"⚡ {dt:.2f}s — Model: MN_Whisper_Small_CT2 ({compute_type})")
    else:
        st.warning("⚠️ No speech detected or audio too quiet.")

# ---------- EXEC ----------
if audio_bytes is not None:
    handle_audio(audio_bytes)

if st.button("🔁 Retry last audio"):
    if st.session_state.get("last_audio_bytes"):
        handle_audio(st.session_state["last_audio_bytes"])
    else:
        st.info("No previous audio to retry yet.")

if st.session_state["last_text"]:
    st.markdown("---")
    st.markdown(
        f"<p style='font-size:1.1rem;color:#444;'>🗣️ <b>Last recognized text:</b> "
        f"{st.session_state['last_text']}</p>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#666;'>Developed by <b>Gankhuyag Mambaryenchin</b><br>"
    "Fine-tuned Whisper Model — Mongolian Fast-Whisper (v3.3 Audio-Verified)</p>",
    unsafe_allow_html=True)
