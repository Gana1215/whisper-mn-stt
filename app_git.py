# ===============================================
# 🎙️ Mongolian Fast-Whisper STT (v3.3 — Universal Recorder, Fully Interactive)
# ✅ Works in iPhone Chrome/Safari, Windows/Mac Chrome
# ✅ Immediate transcription after Stop
# ===============================================

import streamlit as st
import torch, logging, sys, io, time, tempfile, platform, concurrent.futures, inspect, os, base64
import numpy as np, soundfile as sf, scipy.signal, av
from faster_whisper import WhisperModel
from streamlit_javascript import st_javascript  # ✅ Make sure installed

# ---------- CPU guards ----------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try: torch.set_num_interop_threads(1)
    except Exception: pass

# ---------- LOGGER ----------
logger = logging.getLogger("trace")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s] TRACE: %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
def trace(msg: str): logger.info(msg); st.caption(f"🧭 {msg}")

# ---------- PAGE ----------
st.set_page_config(page_title="🎙️ Mongolian Fast-Whisper STT", page_icon="🎧", layout="centered")
st.markdown("""
<style>
html,body,[class*="css"]{font-family:'Noto Sans MN',sans-serif;background:radial-gradient(circle at top left,#f0f2f6,#dfe4ea);}
h1{background:-webkit-linear-gradient(45deg,#0f4c81,#1f8ac0);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800;text-align:center;}
.recorder-btn{background:#0f4c81;color:white;border:none;border-radius:50%;
width:100px;height:100px;box-shadow:0 4px 10px rgba(0,0,0,0.25);
font-size:1.2rem;transition:all 0.3s ease-in-out;}
.recorder-btn.recording{background:#d32f2f;box-shadow:0 0 20px #ff4d4d;}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1>🎙️ Mongolian Fast-Whisper STT</h1>", unsafe_allow_html=True)
st.caption("Universal Recorder Edition — Fine-tuned Whisper Small CT2")

# ---------- MODEL ----------
system, processor = platform.system().lower(), platform.processor().lower()
compute_type = "float32" if ("darwin" in system and "apple" in processor) else "int8"

@st.cache_resource(show_spinner=False)
def load_model():
    trace("Loading Whisper model…")
    return WhisperModel("gana1215/MN_Whisper_Small_CT2", device="cpu", compute_type=compute_type)

with st.spinner("🔁 Loading Whisper model…"):
    model = load_model()
st.success("✅ Model loaded successfully!")

# ---------- Helpers ----------
def decode_webm_to_float32_mono_16k(webm_bytes: bytes):
    container = av.open(io.BytesIO(webm_bytes))
    stream = next(s for s in container.streams if s.type == "audio")
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            frame = resampler.resample(frame)
            arr = frame.to_ndarray()
            if arr.ndim > 1: arr = np.mean(arr, axis=0)
            chunks.append(arr.astype(np.float32) / 32768.0)
    container.close()
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32), 16000

def write_temp_wav(data, sr):
    fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    sf.write(path, data, sr)
    return path

def safe_transcribe(wav_path):
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(model.transcribe, wav_path, language="mn", beam_size=1, vad_filter=False)
            return fut.result(timeout=30)
    except Exception as e:
        st.error(f"❌ Transcription failed: {e}")
        return [], None

# ---------- Recorder JS ----------
st.markdown("""
<script>
let mediaRecorder, chunks = [];
async function toggleRecord() {
  const btn = document.getElementById("recbtn");
  if(!mediaRecorder || mediaRecorder.state==='inactive') {
    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    chunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => chunks.push(e.data);
    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunks, {type:'audio/webm'});
      const arrayBuffer = await blob.arrayBuffer();
      const base64String = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
      window.parent.postMessage({type:'AUDIO_DATA', data: base64String}, "*");
      btn.classList.remove("recording");
      btn.innerText = "🎙 Start";
    };
    mediaRecorder.start();
    btn.classList.add("recording");
    btn.innerText = "⏹ Stop";
  } else {
    mediaRecorder.stop();
  }
}
</script>
<button id="recbtn" class="recorder-btn" onclick="toggleRecord()">🎙 Start</button>
""", unsafe_allow_html=True)

# ---------- Capture JS data ----------
audio_base64 = st_javascript("""new Promise((resolve)=>{
  window.addEventListener("message",(event)=>{
    if(event.data.type==="AUDIO_DATA"){resolve(event.data.data);}
  });
});""")

# ---------- Process after STOP ----------
if audio_base64:
    st.info("⏳ Processing your voice…")
    try:
        audio_bytes = base64.b64decode(audio_base64)
        data, sr = decode_webm_to_float32_mono_16k(audio_bytes)
        tmp = write_temp_wav(data, sr)
        segments, info = safe_transcribe(tmp)
        os.unlink(tmp)
        text = " ".join([s.text.strip() for s in segments if getattr(s,"text","").strip()]) if segments else ""
        if text:
            st.success("✅ Recognition complete!")
            st.markdown(f"<div style='padding:1rem;background:#f8f9fa;border-radius:12px;font-size:1.3rem;color:#111;'>{text}</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ No speech detected.")
    except Exception as e:
        st.error(f"Error: {e}")

# ---------- Footer ----------
st.markdown("---")
st.markdown("<p style='text-align:center;color:#666;'>Developed by <b>Gankhuyag Mambaryenchin</b><br>Fine-tuned Whisper Model — v3.3 Universal Recorder</p>", unsafe_allow_html=True)
