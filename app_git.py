# ===============================================
# 🎙️ Mongolian Fast-Whisper STT (v3.0 — Custom Recorder Edition)
# ✅ Works on iPhone Safari/Chrome, Android, Mac, Windows
# ✅ No st-audiorec or st.audio_input dependency
# ===============================================

import streamlit as st
import base64, io, os, soundfile as sf, numpy as np, time, tempfile, inspect, concurrent.futures
from faster_whisper import WhisperModel

# ---------------------- UI SETUP ----------------------
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
button{background:linear-gradient(90deg,#0f4c81,#1f8ac0);color:white;
border:none;padding:0.8rem 1.5rem;border-radius:8px;font-size:1rem;cursor:pointer;}
button:disabled{opacity:0.5;cursor:not-allowed;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎙️ Mongolian Fast-Whisper STT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>(Custom Recorder Edition — Universal Browser Safe)</p>", unsafe_allow_html=True)

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource(show_spinner=True)
def load_model():
    return WhisperModel("gana1215/MN_Whisper_Small_CT2", device="cpu", compute_type="int8")

model = load_model()
st.success("✅ Model loaded successfully!")

# ---------------------- JS RECORDER ----------------------
st.markdown("### 🎤 Record your voice below and click Stop to transcribe:")

st.components.v1.html("""
<div style='text-align:center'>
  <button id="recBtn">🎙️ Start Recording</button>
  <button id="stopBtn" disabled>⏹ Stop</button>
  <p id="status" style='color:#555;font-size:1.1rem'></p>
</div>
<script>
let recBtn=document.getElementById('recBtn');
let stopBtn=document.getElementById('stopBtn');
let status=document.getElementById('status');
let mediaRecorder,audioChunks=[];

recBtn.onclick=async()=>{
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    audioChunks=[];
    mediaRecorder=new MediaRecorder(stream);
    mediaRecorder.start();
    recBtn.disabled=true; stopBtn.disabled=false;
    status.innerText='🎙️ Recording... speak now';
    mediaRecorder.ondataavailable=e=>audioChunks.push(e.data);
  }catch(err){
    status.innerText='❌ Mic access denied.';
  }
};

stopBtn.onclick=()=>{
  mediaRecorder.stop();
  recBtn.disabled=false; stopBtn.disabled=true;
  status.innerText='⏳ Processing...';
  mediaRecorder.onstop=()=>{
    const blob=new Blob(audioChunks,{type:'audio/webm'});
    blob.arrayBuffer().then(buf=>{
      const base64=btoa(String.fromCharCode(...new Uint8Array(buf)));
      window.parent.postMessage({type:'AUDIO_B64',data:base64},'*');
    });
  };
};
</script>
""", height=250)

# ---------------------- JS MESSAGE HANDLER ----------------------
st.markdown("""
<script>
window.addEventListener('message',(e)=>{
  if(e.data.type==='AUDIO_B64'){
    window.parent.postMessage({type:'streamlit:setComponentValue',value:e.data.data},'*');
  }
});
</script>
""", unsafe_allow_html=True)

# Listen for the base64 data
audio_b64 = st.experimental_get_query_params().get("audio_b64")
if "_REC_" in st.session_state:
    audio_b64 = st.session_state["_REC_"]

# ---------------------- DECODE + TRANSCRIBE ----------------------
def transcribe_bytes(b64data: str):
    audio_bytes = base64.b64decode(b64data)
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if data.ndim > 1: data = np.mean(data, axis=1)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, data, sr)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(model.transcribe, tmp.name, language="mn", beam_size=1)
        segments, info = future.result(timeout=40)
    os.unlink(tmp.name)
    text = " ".join([s.text.strip() for s in segments if getattr(s, "text", "").strip()])
    return text

# ---------------------- PROCESS FLOW ----------------------
if audio_b64:
    try:
        st.info("⏳ Recognizing Mongolian speech...")
        text = transcribe_bytes(audio_b64[0])
        if text.strip():
            st.success("✅ Recognition complete!")
            st.markdown(
                f"<div style='padding:1rem;background:#f8f9fa;border-radius:12px;"
                f"font-size:1.3rem;color:#111;'>{text}</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ No speech detected.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#666;'>Developed by <b>Gankhuyag Mambaryenchin</b><br>"
    "Fine-tuned Whisper Model — Mongolian Fast-Whisper (Custom Recorder Edition)</p>",
    unsafe_allow_html=True,
)
