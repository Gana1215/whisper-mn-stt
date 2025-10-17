# ===============================================
# 🎙️ Mongolian Fast-Whisper STT (v2.9.4 — Trace Edition)
# ✅ Auto Recorder Refresh • Single Result Render • Compatibility-safe transcription
# ===============================================

import streamlit as st
import torch, logging, sys, io, time, tempfile, platform, concurrent.futures, inspect
import numpy as np, soundfile as sf, scipy.signal, av
from faster_whisper import WhisperModel
import os

# ---------- CPU & threading guards ----------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

# ---------- TRACE LOGGER ----------
logger = logging.getLogger("trace")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s] TRACE: %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def trace(msg: str):
    """Display trace in both UI and backend logs."""
    st.caption(f"🧭 {msg}")
    logger.info(msg)

# ---------- PAGE STYLING ----------
st.set_page_config(page_title="🎙️ Mongolian Fast-Whisper STT", page_icon="🎧", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+MN&display=swap');
html,body,[class*="css"]{font-family:'Noto Sans MN',sans-serif;background:radial-gradient(circle at top left,#f0f2f6,#dfe4ea);color:#222;}
h1{background:-webkit-linear-gradient(45deg,#0f4c81,#1f8ac0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800;text-align:center;margin-bottom:0.3rem;}
.subtitle{text-align:center;font-size:1.1rem;color:#555;margin-bottom:1rem;font-style:italic;}
[data-testid="stAudioInput"]{margin-top:1.2rem;margin-bottom:1.5rem;display:flex;justify-content:center;}
button[data-testid="stAudioInput__record"],button[data-testid="stAudioInput__stop"]{
transform:scale(1.5);color:white !important;border-radius:50%;border:none;width:80px !important;height:80px !important;font-size:1.1rem;
box-shadow:0 4px 10px rgba(0,0,0,0.25);}
button[data-testid="stAudioInput__record"]{background:linear-gradient(90deg,#0f4c81,#1f8ac0);}
button[data-testid="stAudioInput__stop"]{background:linear-gradient(90deg,#d32f2f,#f44336);}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎙️ Mongolian Fast-Whisper STT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>(Anti-Hallucination Edition — Trace & Auto Refresh)</p>", unsafe_allow_html=True)
st.caption("⚡ Fine-tuned Mongolian Whisper model with full trace diagnostics and automatic recorder refresh")

# ---------- MODEL LOAD ----------
system, processor = platform.system().lower(), platform.processor().lower()
compute_type = "float32" if ("darwin" in system and "apple" in processor) else "int8"

@st.cache_resource(show_spinner=False)
def load_model():
    trace("Loading WhisperModel (CT2 backend)...")
    return WhisperModel("gana1215/MN_Whisper_Small_CT2", device="cpu", compute_type=compute_type)

with st.spinner("🔁 Loading Whisper model..."):
    model = load_model()
st.success("✅ Model loaded successfully!")

# ---------- TRANSCRIBE HELPER ----------
def transcribe_compat(model, path, **kwargs):
    """Handles both show_progress/log_progress versions."""
    sig = inspect.signature(model.transcribe)
    if "show_progress" in sig.parameters:
        kwargs["show_progress"] = False
    elif "log_progress" in sig.parameters:
        kwargs["log_progress"] = False
    return model.transcribe(path, **kwargs)

# ---------- AUDIO HELPERS ----------
def decode_webm_to_float32_mono_16k(webm_bytes: bytes):
    trace("Stage 2: Opening WebM via PyAV...")
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
        trace("Stage 2: No audio frames decoded (empty WebM).")
        return np.zeros((0,), dtype=np.float32), 16000
    trace("Stage 2: Decode successful.")
    return np.concatenate(chunks), 16000

def ensure_mono_16k(data: np.ndarray, sr: int):
    trace(f"Stage 2: Resampling to 16 kHz (current sr={sr})...")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != 16000:
        data = scipy.signal.resample_poly(data, 16000, sr)
        sr = 16000
    return np.nan_to_num(data.astype(np.float32)), sr

def write_temp_wav(data: np.ndarray, sr: int):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp, data, sr)
        trace(f"Stage 2: Temporary WAV written → {tmp.name}")
        return tmp.name

# ---------- SAFE TRANSCRIBE ----------
def safe_transcribe(wav_path: str):
    trace("Stage 3: Starting model.transcribe()…")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                transcribe_compat,
                model,
                wav_path,
                language="mn",
                beam_size=1,
                vad_filter=True,
                suppress_tokens=[-1],
                condition_on_previous_text=False,
                word_timestamps=False,
                temperature=0.0,
            )
            return fut.result(timeout=40)
    except concurrent.futures.TimeoutError:
        trace("Stage 3: TIMEOUT — transcription exceeded 40 s.")
        st.warning("⏳ Timeout — please retry with a shorter clip.")
        return [], None
    except Exception as e:
        trace(f"Stage 3 ERROR: {e}")
        st.error(f"❌ Model error: {e}")
        return [], None
    finally:
        trace("Stage 3: Exited transcription thread.")

# ---------- WARM-UP ----------
if "warmup_done" not in st.session_state:
    st.session_state["warmup_done"] = True
    try:
        trace("Running cold-start warm-up (0.4 s silent audio)…")
        sr = 16000
        warm = np.zeros(int(0.4 * sr), dtype=np.float32)
        path = write_temp_wav(warm, sr)
        transcribe_compat(model, path, language="mn", beam_size=1)
        trace("Warm-up done.")
    except Exception as e:
        trace(f"Warm-up failed: {e}")

# ---------- MAIN UI ----------
st.subheader("🎤 Record your voice below")
st.write("Click the mic icon, speak in Mongolian, then click stop to transcribe:")
st.caption("🟢 Note: The first recording may show 'An error occurred' as your browser initializes the mic. Simply click Start again.")

# Ensure UI area cleared before new text (prevents double render)
if "last_text" not in st.session_state:
    st.session_state["last_text"] = ""

# Unique key ensures new widget instance for each recording
audio_file = st.audio_input("🎙️ Start recording", key=f"rec_{int(time.time())}")

def handle_audio(audio_bytes: bytes, mime: str):
    trace("Stage 1: Received audio from recorder_input.")
    if not audio_bytes or len(audio_bytes) < 800:
        trace("Stage 1: Empty or too-short buffer (<800 B).")
        st.warning("🎙️ Microphone initializing — please click Start again.")
        return
    time.sleep(0.3)

    try:
        if "webm" in (mime or ""):
            data, sr = decode_webm_to_float32_mono_16k(audio_bytes)
        else:
            trace("Stage 2: Reading as WAV/PCM via soundfile…")
            data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            data, sr = ensure_mono_16k(data, sr)
    except Exception as e:
        trace(f"Stage 2 ERROR: {e}")
        st.error(f"Decode failed: {e}")
        return

    if data.size == 0 or len(data) < sr * 0.3:
        trace("Stage 2: Audio buffer empty or <0.3 s.")
        st.warning("⚠️ Recording too short.")
        return

    tmp = write_temp_wav(data, sr)
    st.info("⏳ Recognizing Mongolian speech…")

    t0 = time.time()
    segments, info = safe_transcribe(tmp)
    dt = time.time() - t0

    text = " ".join([s.text.strip() for s in segments if getattr(s, "text", "").strip()]) if segments else ""

    st.session_state["last_text"] = text  # ✅ store latest text

    if text:
        trace("Stage 4: Recognition successful.")
        st.success("✅ Recognition complete!")
        st.markdown(
            f"<div style='padding:1rem;background:#f8f9fa;border-radius:12px;font-size:1.3rem;color:#111;'>{text}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"⚡ {dt:.2f}s — Model: MN_Whisper_Small_CT2 ({compute_type})")
    else:
        trace("Stage 4: No speech detected.")
        st.warning("⚠️ No speech detected.")

# ---------- EXECUTION FLOW ----------
if audio_file is not None:
    trace("Stage 0: Audio input triggered.")
    st.session_state["last_audio_bytes"] = None  # clear
    try:
        audio_bytes = audio_file.read()
        if audio_bytes:
            st.session_state["last_audio_bytes"] = (audio_bytes, audio_file.type)
            handle_audio(audio_bytes, audio_file.type)
        else:
            trace("Stage 0: Empty buffer, skipping.")
            st.warning("🎙️ Empty recording — please retry.")
    except Exception as e:
        trace(f"Top-level ERROR: {e}")
        st.error(f"❌ Unexpected error: {e}")
else:
    trace("Stage 0: Waiting for recording input.")
    st.info("⏺️ Waiting for you to record…")

# Retry button always available
if st.button("🔁 Retry last audio"):
    trace("Retry button clicked.")
    if st.session_state.get("last_audio_bytes"):
        audio_bytes, mime = st.session_state["last_audio_bytes"]
        handle_audio(audio_bytes, mime)
    else:
        st.info("No previous audio to retry yet.")

# Show last recognized text only once
if st.session_state["last_text"]:
    st.markdown("---")
    st.markdown(
        f"<p style='font-size:1.1rem;color:#444;'>🗣️ <b>Last recognized text:</b> {st.session_state['last_text']}</p>",
        unsafe_allow_html=True,
    )

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#666;'>Developed by <b>Gankhuyag Mambaryenchin</b><br>"
    "Fine-tuned Whisper Model — Mongolian Fast-Whisper (Trace Edition v2.9.4)</p>",
    unsafe_allow_html=True,
)
