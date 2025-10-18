# ===============================================
# 🎙️ Mongolian Fast-Whisper STT (v4.3.6 — Safe Recorder + Auto-Initialize)
# ✅ Fully compatible with Streamlit Cloud, iPhone Safari, Chrome, Android, macOS, Windows
# ✅ Fine-tuned MN_Whisper_Small_CT2 model from Hugging Face
# ===============================================

import os, io, tempfile, time, platform, concurrent.futures, json, logging, sys, inspect
import numpy as np
import soundfile as sf
import av
import streamlit as st
import streamlit.components.v1 as components
from faster_whisper import WhisperModel
import torch

# ---------- THREADING GUARDS ----------
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
handler.setFormatter(logging.Formatter("[%(asctime)s] TRACE: %(message)s", "%H:%M:%S"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def trace(msg: str):
    st.caption(f"🧭 {msg}")
    logger.info(msg)

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="🎙️ Mongolian Fast-Whisper STT", page_icon="🎧", layout="centered")

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
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  font-weight:800;text-align:center;margin-bottom:0.3rem;
}
.subtitle{text-align:center;font-size:1.05rem;color:#555;margin-bottom:1rem;font-style:italic;}
.result{padding:1rem;background:#f8f9fa;border-radius:12px;font-size:1.2rem;color:#111;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎙️ Mongolian Fast-Whisper STT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>(v4.3.6 — Safe Recorder + Auto-Initialize)</p>", unsafe_allow_html=True)
st.caption("⚡ Fine-tuned Mongolian Whisper model — mobile compatible & cloud optimized.")

# ---------- MODEL LOAD ----------
system, processor = platform.system().lower(), platform.processor().lower()
compute_type = "float32" if ("darwin" in system and "apple" in processor) else "int8"

@st.cache_resource(show_spinner=False)
def load_model():
    trace("Loading WhisperModel (CT2 backend, remote allowed)…")
    return WhisperModel(
        "gana1215/MN_Whisper_Small_CT2",
        device="cpu",
        compute_type=compute_type,
        local_files_only=False
    )

with st.spinner("🔁 Loading Whisper model (first load may take ~1 min)…"):
    model = load_model()
st.success("✅ Model loaded successfully!")

# ---------- TRANSCRIBE HELPERS ----------
def transcribe_compat(model, path, **kwargs):
    sig = inspect.signature(model.transcribe)
    if "show_progress" in sig.parameters:
        kwargs["show_progress"] = False
    elif "log_progress" in sig.parameters:
        kwargs["log_progress"] = False
    return model.transcribe(path, **kwargs)

def safe_transcribe(wav_path: str):
    trace("Stage 3: Starting model.transcribe()…")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                transcribe_compat,
                model, wav_path,
                language="mn",
                beam_size=1,
                vad_filter=True,
                suppress_tokens=[-1],
                condition_on_previous_text=False,
                word_timestamps=False,
                temperature=0.0,
            )
            return fut.result(timeout=45)
    except concurrent.futures.TimeoutError:
        st.warning("⏳ Timeout — please retry with a shorter clip.")
        return [], None
    except Exception as e:
        st.error(f"❌ Model error: {e}")
        return [], None

# ---------- AUDIO HELPERS ----------
def decode_webm_to_float32_mono_16k(webm_bytes: bytes):
    trace("Stage 2: Decoding WebM/Opus via PyAV…")
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

def write_temp_wav(data: np.ndarray, sr: int):
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, data, sr)
    return path

# ---------- STATE ----------
if "last_text" not in st.session_state:
    st.session_state["last_text"] = ""
if "last_audio_bytes" not in st.session_state:
    st.session_state["last_audio_bytes"] = None
if "last_mime" not in st.session_state:
    st.session_state["last_mime"] = "audio/webm"

# ---------- CUSTOM RECORDER (v4.3.6) ----------
st.markdown("### 🎙️ Voice Recorder")
st.caption("If you don’t see the mic button, click below to initialize the recorder.")

if st.button("🔄 Initialize Recorder"):
    st.rerun()

html_code = """
<div style="display:flex;flex-direction:column;align-items:center;gap:8px">
  <button id="recbtn" style="
      background:#0f4c81;color:white;border:none;border-radius:50%;
      width:110px;height:110px;font-size:1.1rem;box-shadow:0 4px 12px rgba(0,0,0,0.3);
      transition:all 0.25s ease;">🎙️ Start</button>
  <div id="status" style="color:#555;font-size:0.95rem;">Tap Start, speak, then tap Stop</div>
</div>
<script>
let rec,chunks=[];
const btn=document.getElementById("recbtn");
const status=document.getElementById("status");
window.parent.postMessage({type:'streamlit:setComponentValue',value:""},'*');

async function startRec(){
 try{
   const stream=await navigator.mediaDevices.getUserMedia({audio:true});
   rec=new MediaRecorder(stream);
   chunks=[];
   rec.ondataavailable=e=>{if(e.data.size>0)chunks.push(e.data)};
   rec.onstop=async()=>{
     const blob=new Blob(chunks,{type:'audio/webm'});
     status.textContent='Processing...';
     const ab=await blob.arrayBuffer();
     const bytes=Array.from(new Uint8Array(ab));
     const json=JSON.stringify({audio:bytes,mime:blob.type||'audio/webm'});
     window.parent.postMessage({type:'streamlit:setComponentValue',value:json},'*');
     setTimeout(()=>window.parent.postMessage({type:'streamlit:setComponentValue',value:json},'*'),50);
     status.textContent='✅ Audio sent to model.';
   };
   rec.start();
   btn.textContent='⏹ Stop';
   btn.style.background='#d32f2f';
   status.textContent='Recording...';
 }catch(err){status.textContent='❌ Mic permission denied or unavailable';}
}

function stopRec(){
 if(rec&&rec.state!=='inactive'){
   rec.stop();
   btn.textContent='🎙️ Start';
   btn.style.background='#0f4c81';
   status.textContent='Stopping...';
 }
}
btn.addEventListener('click',()=>{if(!rec||rec.state==='inactive')startRec();else stopRec();});
</script>
"""

try:
    rec = components.html(html_code, height=250, scrolling=False, key="recorder_v436")
except TypeError:
    st.warning("⚙️ Recorder sandbox initializing — click '🔄 Initialize Recorder' if nothing appears.")
    rec = ""
except Exception as e:
    st.warning(f"⚠️ Recorder sandbox issue: {e}")
    rec = ""

# ---------- PIPELINE ----------
if isinstance(rec, str) and len(rec) > 10:
    try:
        payload = json.loads(rec)
        audio_bytes = bytes(payload["audio"])
        mime = payload.get("mime", "audio/webm")
        st.session_state["last_audio_bytes"] = audio_bytes
        st.session_state["last_mime"] = mime

        data, sr = decode_webm_to_float32_mono_16k(audio_bytes)
        if data.size < int(sr * 0.25):
            st.warning("⚠️ Too short or silent — please try again.")
        else:
            st.info("⏳ Recognizing Mongolian speech…")
            t0 = time.time()
            tmp = write_temp_wav(data, sr)
            segments, info = safe_transcribe(tmp)
            try: os.unlink(tmp)
            except Exception: pass
            dt = time.time() - t0
            text = " ".join([s.text.strip() for s in segments if getattr(s, "text", "").strip()]) if segments else ""
            st.session_state["last_text"] = text
            if text:
                st.success("✅ Recognition complete!")
                st.markdown(f"<div class='result'>{text}</div>", unsafe_allow_html=True)
                st.caption(f"⚡ {dt:.2f}s — MN_Whisper_Small_CT2 ({compute_type})")
            else:
                st.warning("⚠️ No speech detected.")
    except Exception as e:
        st.error(f"❌ Recorder error: {e}")

# ---------- RETRY + DOWNLOAD ----------
col1, col2 = st.columns(2)
with col1:
    if st.button("🔁 Retry last audio", use_container_width=True):
        if st.session_state.get("last_audio_bytes"):
            raw = st.session_state["last_audio_bytes"]
            data, sr = decode_webm_to_float32_mono_16k(raw)
            if data.size < int(sr * 0.25):
                st.warning("⚠️ Last audio too short.")
            else:
                st.info("⏳ Re-transcribing last recording…")
                t0 = time.time()
                tmp = write_temp_wav(data, sr)
                segments, info = safe_transcribe(tmp)
                try: os.unlink(tmp)
                except Exception: pass
                dt = time.time() - t0
                text = " ".join([s.text.strip() for s in segments if getattr(s, "text", "").strip()]) if segments else ""
                st.session_state["last_text"] = text
                if text:
                    st.success("✅ Recognition complete!")
                    st.markdown(f"<div class='result'>{text}</div>", unsafe_allow_html=True)
                    st.caption(f"⚡ {dt:.2f}s — Retried")
                else:
                    st.warning("⚠️ No speech detected.")
        else:
            st.info("No previous audio to retry yet.")
with col2:
    if st.session_state.get("last_audio_bytes"):
        st.download_button(
            "⬇️ Download last audio",
            data=st.session_state["last_audio_bytes"],
            file_name="recording.webm",
            mime=st.session_state.get("last_mime", "audio/webm"),
            use_container_width=True
        )
    else:
        st.button("⬇️ Download last audio", disabled=True, use_container_width=True)

# ---------- FOOTER ----------
if st.session_state.get("last_text"):
    st.markdown("---")
    st.markdown(
        f"<p style='font-size:1.05rem;color:#444;'>🗣️ <b>Last recognized text:</b> {st.session_state['last_text']}</p>",
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#666;'>Developed by <b>Gankhuyag Mambaryenchin</b><br>"
    "Fine-tuned Whisper Model — Mongolian Fast-Whisper (v4.3.6)</p>",
    unsafe_allow_html=True
)
