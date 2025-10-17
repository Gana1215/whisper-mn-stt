# ===============================================
# 🎙️ Mongolian Fast-Whisper STT (v2.5 — Cloud-Stable + Auto Warm-Up)
# ✅ PyAV WebM/Opus, safe I/O, guarded inference, first-click warm-up fix
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

# ---------- Page Setup ----------
st.set_page_config(page_title="🎙️ Mongolian Fast-Whisper STT", page_icon="🎧", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+MN&display=swap');
html,body,[class*="css"]{font-family:'Noto Sans MN',sans-serif;background:radial-gradient(circle at top left,#f0f2f6,#dfe4ea);color:#222;}
h1{background:-webkit-linear-gradient(45deg,#0f4c81,#1f8ac0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800;text-align:center;margin-bottom:0.3rem;}
.subtitle{text-align:center;font-size:1.1rem;color:#555;margin-bottom:1rem;font-style:italic;}
div.stButton>button:first-child{background:linear-gradient(90deg,#0f4c81,#1f8ac0);color:white;font-weight:bold;border-radius:12px;padding:0.6rem 1.2rem;border:none;}
.stSuccess,.stInfo,.stWarning,.stError{border-radius:10px;}
[data-testid="stAudioInput"]{margin-top:1.2rem;margin-bottom:1.5rem;display:flex;justify-content:center;}
button[data-testid="stAudioInput__record"]{transform:scale(1.5);background:linear-gradient(90deg,#0f4c81,#1f8ac0);color:white !important;border-radius:50%;border:none;width:80px !important;height:80px !important;font-size:1.1rem;box-shadow:0 4px 10px rgba(0,0,0,0.25);}
button[data-testid="stAudioInput__stop"]{transform:scale(1.5);background:linear-gradient(90deg,#d32f2f,#f44336);color:white !important;border-radius:50%;border:none;width:80px !important;height:80px !important;font-size:1.1rem;box-shadow:0 4px 10px rgba(0,0,0,0.25);}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎙️ Mongolian Fast-Whisper STT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>(Anti-Hallucination Edition — Cloud-Stable)</p>", unsafe_allow_html=True)
st.caption("⚡ Fine-tuned Mongolian Whisper model with stable cloud inference")

# ---------- Model Loading ----------
system = platform.system().lower()
processor = platform.processor().lower()
compute_type = "float32" if ("darwin" in system and "apple" in processor) else "int8"

@st.cache_resource(show_spinner=False)
def load_model():
    repo_id = "gana1215/MN_Whisper_Small_CT2"
    return WhisperModel(repo_id, device="cpu", compute_type=compute_type)

with st.spinner("🔁 Loading Whisper model..."):
    model = load_model()
st.success("✅ Model loaded successfully! Ready to transcribe your voice.")

# ---------- Helpers ----------
def decode_webm_to_float32_mono_16k(webm_bytes: bytes):
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

def write_temp_wav(data: np.ndarray, sr: int) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp, data, sr)
        return tmp.name

def safe_transcribe(wav_path: str):
    """Guarded Faster-Whisper call tuned for Streamlit Cloud compatibility."""
    return model.transcribe(
        wav_path,
        language="mn",
        beam_size=1,
        vad_filter=True,
        suppress_tokens="-1",
        condition_on_previous_text=False,
        word_timestamps=False
    )

# ---------- One-Time Auto Warm-Up ----------
if "warmup_done" not in st.session_state:
    try:
        sr = 16000
        warm = (np.zeros(int(0.4 * sr), dtype=np.float32) + 1e-7)
        warm_path = write_temp_wav(warm, sr)
        _ = safe_transcribe(warm_path)
    except Exception as e:
        st.caption(f"🛠 Warm-up note: {e}")
    finally:
        st.session_state["warmup_done"] = True

# ---------- UI: Recording ----------
st.subheader("🎤 Record your voice below")
st.write("Click the mic icon, speak in Mongolian, then click stop to transcribe:")
debug = st.toggle("Show debug info", value=False)

audio_file = st.audio_input("🎙️ Start recording")
if "last_audio_bytes" not in st.session_state:
    st.session_state["last_audio_bytes"] = None

def handle_audio_bytes(audio_bytes: bytes, mime: str):
    if not audio_bytes or len(audio_bytes) < 500:
        st.warning("🕒 Initial microphone warm-up… please record again.")
        return
    time.sleep(0.3)

    try:
        if mime and "webm" in mime:
            if debug: st.caption("Decoder: PyAV (WebM/Opus → 16 kHz mono)")
            data, sr = decode_webm_to_float32_mono_16k(audio_bytes)
        else:
            if debug: st.caption("Decoder: SoundFile (WAV/PCM)")
            data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            data, sr = ensure_mono_16k(data, sr)
    except Exception as e:
        st.error(f"❌ Audio decode failed: {e}")
        if debug: st.exception(e)
        return

    if len(data) < sr * 0.3:
        st.warning("⚠️ Recording too short (<0.3 s). Please speak a bit longer.")
        return

    if debug: st.caption(f"📊 Decoded: shape={data.shape}, sr={sr} Hz")
    wav_path = write_temp_wav(data, sr)

    if st.session_state.get("is_transcribing", False):
        st.warning("⚙️ Already transcribing… please wait.")
        return
    st.session_state["is_transcribing"] = True

    st.info("⏳ Recognizing your Mongolian speech…")
    t0 = time.time()
    try:
        segments, info = safe_transcribe(wav_path)
    except Exception as e:
        st.error("❌ Transcription failed. Please record again.")
        if debug: st.exception(e)
        st.session_state["is_transcribing"] = False
        return

    text = " ".join([s.text.strip() for s in segments if getattr(s, "text", "").strip()]) if segments else ""
    dt = time.time() - t0
    st.session_state["is_transcribing"] = False

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

# ----- Live Recording Flow -----
if audio_file is not None:
    st.success(f"🎧 Recorded audio received — {audio_file.size} bytes")
    st.caption(f"📁 MIME type: {audio_file.type}")
    try:
        audio_bytes = audio_file.read()
        st.session_state["last_audio_bytes"] = (audio_bytes, audio_file.type)
        handle_audio_bytes(audio_bytes, audio_file.type)
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        if debug: st.exception(e)
else:
    st.info("⏺️ Waiting for you to record…")
    st.caption("💡 Tip: On first click, allow mic access. If nothing transcribes, record once more.")

# ---------- Retry + File Upload ----------
col1, col2 = st.columns([1,3])
with col1:
    if st.button("🔁 Retry last audio"):
        if st.session_state.get("last_audio_bytes"):
            audio_bytes, mime = st.session_state["last_audio_bytes"]
            handle_audio_bytes(audio_bytes, mime)
        else:
            st.info("No previous audio to retry yet.")

with col2:
    upload = st.file_uploader("📂 Or upload .wav / .mp3 / .webm file", type=["wav","mp3","webm"], label_visibility="collapsed")
    if upload is not None:
        st.info(f"📥 Uploaded: {upload.name}")
        audio_bytes = upload.read()
        handle_audio_bytes(audio_bytes, upload.type or upload.name.split(".")[-1])

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#666;'>Developed by <b>Gankhuyag Mambaryenchin</b><br>"
    "Fine-tuned Whisper Model — Mongolian Fast-Whisper (Anti-Hallucination Edition v2.5)</p>",
    unsafe_allow_html=True
)
