# ===============================================
# 🎙️ Mongolian Fast-Whisper STT (v3.6 — Cloud Recorder Fix)
# ✅ Works on Streamlit Cloud + iPhone Safari/Chrome
# ✅ Recorder initialized after DOM ready
# ===============================================

import streamlit as st
import torch, io, os, sys, time, tempfile, logging, concurrent.futures, base64, platform
import numpy as np, soundfile as sf, scipy.signal, av
from faster_whisper import WhisperModel
from streamlit_javascript import st_javascript

# ---------- Setup ----------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try: torch.set_num_interop_threads(1)
    except Exception: pass

st.set_page_config(page_title="🎙️ Mongolian Fast-Whisper STT", page_icon="🎧", layout="centered")

st.markdown("""
<style>
html,body,[class*="css"]{
  font-family:'Noto Sans MN',sans-serif;
  background:radial-gradient(circle at top left,#f0f2f6,#dfe4ea);
  color:#222;
}
h1{
  background:-webkit-linear-gradient(45deg,#0f4c81,#1f8ac0);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  font-weight:800;text-align:center;
}
button.recorder {
  background:#0f4c81;color:white;border:none;border-radius:50%;
  width:100px;height:100px;font-size:1.1rem;
  box-shadow:0 4px 10px rgba(0,0,0,0.25);
  transition:all .25s ease-in-out;
}
button.recorder.recording {
  background:#d32f2f;box-shadow:0 0 20px #ff4d4d;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎙️ Mongolian Fast-Whisper STT</h1>", unsafe_allow_html=True)
st.caption("v3.6 — Cloud-safe universal recorder (iPhone + desktop)")

# ---------- Logger ----------
def trace(msg): st.caption(f"🧭 {msg}")

# ---------- Model ----------
system, proc = platform.system().lower(), platform.processor().lower()
compute_type = "float32" if ("darwin" in system and "apple" in proc) else "int8"

@st.cache_resource(show_spinner=False)
def load_model():
    trace("Loading Whisper model (CT2 backend)…")
    return WhisperModel("gana1215/MN_Whisper_Small_CT2", device="cpu", compute_type=compute_type)

with st.spinner("🔁 Loading Whisper model…"):
    model = load_model()
st.success("✅ Model loaded successfully!")

# ---------- Audio utils ----------
def decode_webm_to_float32_mono_16k(webm_bytes: bytes):
    container = av.open(io.BytesIO(webm_bytes))
    stream = next(s for s in container.streams if s.type == "audio")
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks=[]
    for pkt in container.demux(stream):
        for frame in pkt.decode():
            f=resampler.resample(frame)
            arr=f.to_ndarray()
            if arr.ndim>1: arr=np.mean(arr,axis=0)
            chunks.append(arr.astype(np.float32)/32768.0)
    container.close()
    return np.concatenate(chunks) if chunks else np.zeros(0,np.float32),16000

def write_temp_wav(data,sr):
    fd,path=tempfile.mkstemp(suffix=".wav");os.close(fd);sf.write(path,data,sr);return path

def safe_transcribe(path):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut=ex.submit(model.transcribe,path,language="mn",beam_size=1,vad_filter=False)
        return fut.result(timeout=40)

# ---------- Recorder UI + JS ----------
st.markdown("""
<div style="text-align:center;">
  <button id="recbtn" class="recorder">🎙 Start</button>
  <p id="statusText" style="font-style:italic;color:#555;">Ready to record</p>
</div>
<script>
document.addEventListener("DOMContentLoaded",()=>{
  let chunks=[],mediaRecorder;
  const btn=document.getElementById("recbtn");
  const status=document.getElementById("statusText");
  async function startRec(){
    const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    chunks=[];mediaRecorder=new MediaRecorder(stream);
    mediaRecorder.ondataavailable=e=>chunks.push(e.data);
    mediaRecorder.onstop=async()=>{
      const blob=new Blob(chunks,{type:'audio/webm'});
      const buf=await blob.arrayBuffer();
      const b64=btoa(String.fromCharCode(...new Uint8Array(buf)));
      status.innerText="Processing audio…";
      window.streamlitRPC.send(b64);
      btn.classList.remove("recording");
      btn.innerText="🎙 Start";
    };
    mediaRecorder.start();
    btn.classList.add("recording");
    btn.innerText="⏹ Stop";
    status.innerText="Recording…";
    btn.onclick=stopRec;
  }
  function stopRec(){mediaRecorder.stop();status.innerText="Stopped";}
  btn.onclick=startRec;
});
</script>
""", unsafe_allow_html=True)

audio_base64 = st_javascript("""new Promise(resolve=>{
  window.streamlitRPC.onReceive=(data)=>resolve(data);
});""")

# ---------- After recording ----------
if audio_base64:
    st.info("⏳ Transcribing your voice…")
    try:
        audio_bytes=base64.b64decode(audio_base64)
        data,sr=decode_webm_to_float32_mono_16k(audio_bytes)
        tmp=write_temp_wav(data,sr)
        segments,info=safe_transcribe(tmp)
        os.unlink(tmp)
        text=" ".join([s.text.strip() for s in segments if getattr(s,"text","").strip()]) if segments else ""
        if text:
            st.success("✅ Recognition complete!")
            st.markdown(f"<div style='padding:1rem;background:#f8f9fa;border-radius:12px;font-size:1.3rem;color:#111;'>{text}</div>",unsafe_allow_html=True)
        else:
            st.warning("⚠️ No speech detected.")
    except Exception as e:
        st.error(f"❌ {e}")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#666;'>Developed by <b>Gankhuyag Mambaryenchin</b><br>Fine-tuned Whisper Small CT2 — v3.6 Cloud Recorder</p>", unsafe_allow_html=True)
