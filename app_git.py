# ===============================================
# 🎙️ Mongolian Fast-Whisper STT (v3.2 — Universal Custom Recorder)
# ✅ Works in Chrome/Safari on iPhone, Mac, Windows
# ✅ Uses stable local Whisper load from Hugging Face cache
# ===============================================

import streamlit as st
import torch, logging, sys, io, time, tempfile, platform, concurrent.futures, inspect, os, base64
import numpy as np, soundfile as sf, scipy.signal, av
from faster_whisper import WhisperModel

# ---------- CPU & threading guards ----------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

# ---------- LOGGER ----------
logger = logging.getLogger("trace")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s] TRACE: %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
def trace(msg: str):
    st.caption(f"🧭 {msg}")
    logger.info(msg)

# ---------- PAGE ----------
st.set_page_config(page_title="🎙️ Mongolian Fast-Whisper STT", page_icon="🎧", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+MN&display=swap');
html,body,[class*="css"]{font-family:'Noto Sans MN',sans-serif;background:radial-gradient(circle at top left,#f0f2f6,#dfe4ea);}
h1{background:-webkit-linear-gradient(45deg,#0f4c81,#1f8ac0);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800;text-align:center;}
.subtitle{text-align:center;font-size:1.1rem;color:#555;margin-bottom:1rem;font-style:italic;}
.recorder-btn{background:#0f4c81;color:white;border:none;border-radius:50%;width:90px;height:90px;
box-shadow:0 4px 10px rgba(0,0,0,0.25);font-size:1.2rem;}
.recorder-btn.stop{background:#d32f2f;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎙️ Mongolian Fast-Whisper STT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>(Universal Custom Recorder — Stable Whisper Model)</p>", unsafe_allow_html=True)

# ---------- MODEL LOAD ----------
system, processor = platform.system().lower(), platform.processor().lower()
compute_type = "float32" if ("darwin" in system and "apple" in processor) else "int8"

@st.cache_resource(show_spinner=False)
def load_model():
    trace("Loading WhisperModel (CT2 backend)…")
    return WhisperModel("gana1215/MN_Whisper_Small_CT2", device="cpu", compute_type=compute_type)

with st.spinner("🔁 Loading Whisper model…"):
    model = load_model()
st.success("✅ Model loaded successfully!")

# ---------- TRANSCRIBE ----------
def transcribe_compat(model, path, **kwargs):
    sig = inspect.signature(model.transcribe)
    if "show_progress" in sig.parameters:
        kwargs["show_progress"] = False
    elif "log_progress" in sig.parameters:
        kwargs["log_progress"] = False
    return model.transcribe(path, **kwargs)

# ---------- Decode & Helpers ----------
def decode_webm_to_float32_mono_16k(webm_bytes: bytes):
    container = av.open(io.BytesIO(webm_bytes))
    astream = next(s for s in container.streams if s.type == "audio")
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    for packet in container.demux(astream):
        for frame in packet.decode():
            f = resampler.resample(frame)
            arr = f.to_ndarray()
            if arr.ndim > 1: arr = np.mean(arr, axis=0)
            chunks.append(arr.astype(np.float32) / 32768.0)
    container.close()
    if not chunks:
        return np.zeros((0,), dtype=np.float32), 16000
    return np.concatenate(chunks), 16000

def ensure_mono_16k(data: np.ndarray, sr: int):
    if data.ndim > 1: data = np.mean(data, axis=1)
    if sr != 16000: data = scipy.signal.resample_poly(data, 16000, sr); sr = 16000
    return np.nan_to_num(data.astype(np.float32)), sr

def write_temp_wav(data: np.ndarray, sr: int):
    fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    sf.write(path, data, sr)
    return path

# ---------- SAFE TRANSCRIBE ----------
def safe_transcribe(wav_path: str):
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                transcribe_compat, model, wav_path,
                language="mn", beam_size=1,
                vad_filter=True, suppress_tokens=[-1],
                condition_on_previous_text=False, word_timestamps=False,
                temperature=0.0,
            )
            return fut.result(timeout=40)
    except Exception as e:
        st.error(f"❌ Model error: {e}")
        return [], None

# ---------- Custom Recorder (JS) ----------
st.markdown("""
<script>
let mediaRecorder, audioChunks = [];
function startRecording() {
    audioChunks = [];
    navigator.mediaDevices.getUserMedia({audio:true})
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            document.getElementById("recbtn").classList.add("stop");
            document.getElementById("recbtn").innerText = "⏹ Stop";
        });
}
function stopRecording() {
    mediaRecorder.stop();
    mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunks, {type:'audio/webm'});
        const arrayBuffer = await blob.arrayBuffer();
        const base64String = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
        window.parent.postMessage({type:'AUDIO_DATA', data: base64String}, "*");
        document.getElementById("recbtn").classList.remove("stop");
        document.getElementById("recbtn").innerText = "🎙 Start";
    };
}
function toggleRecording() {
    if(!mediaRecorder || mediaRecorder.state==='inactive') startRecording();
    else stopRecording();
}
</script>
<button id="recbtn" class="recorder-btn" onclick="toggleRecording()">🎙 Start</button>
""", unsafe_allow_html=True)

# ---------- Audio handler via message listener ----------
audio_base64 = st.session_state.get("audio_base64", None)
st.markdown("""
<script>
window.addEventListener("message", (event) => {
  if(event.data.type === "AUDIO_DATA"){
      const audioBase64 = event.data.data;
      window.parent.postMessage({type: "STREAMLIT_AUDIO", data: audioBase64}, "*");
  }
});
</script>
""", unsafe_allow_html=True)

# Streamlit component bridge
from streamlit_javascript import st_javascript

try:
    js_data = st_javascript("""new Promise((resolve)=>{
        window.addEventListener("message",(event)=>{
            if(event.data.type==="STREAMLIT_AUDIO"){
                resolve(event.data.data);
            }
        });
    });""")
    if js_data:
        audio_base64 = js_data
        st.session_state["audio_base64"] = audio_base64
except Exception:
    pass

# ---------- Process recorded audio ----------
if audio_base64:
    st.info("⏳ Processing recorded audio…")
    try:
        audio_bytes = base64.b64decode(audio_base64)
        data, sr = decode_webm_to_float32_mono_16k(audio_bytes)
        data, sr = ensure_mono_16k(data, sr)
        tmp = write_temp_wav(data, sr)
        segments, info = safe_transcribe(tmp)
        text = " ".join([s.text.strip() for s in segments if getattr(s,"text","").strip()]) if segments else ""
        os.unlink(tmp)
        if text:
            st.success("✅ Recognition complete!")
            st.markdown(f"<div style='padding:1rem;background:#f8f9fa;border-radius:12px;font-size:1.3rem;color:#111;'>{text}</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ No speech detected.")
    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#666;'>Developed by <b>Gankhuyag Mambaryenchin</b><br>"
    "Fine-tuned Whisper Model — Mongolian Fast-Whisper (v3.2 Universal)</p>",
    unsafe_allow_html=True,
)
