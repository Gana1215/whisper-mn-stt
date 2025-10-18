# ===============================================
# 🎙️ Mongolian Fast-Whisper STT (v4.3.1 — Stable Custom Recorder)
# ✅ Cross-browser (iPhone Safari/Chrome, Android, Desktop)
# ✅ Dual-message bridge fix (Streamlit Cloud safe)
# ✅ Uses gana1215/MN_Whisper_Small_CT2
# ===============================================

import os, io, base64, tempfile, time, platform, concurrent.futures, json
import numpy as np
import soundfile as sf
import av
import streamlit as st
import streamlit.components.v1 as components
from faster_whisper import WhisperModel

# ---------- PAGE SETUP ----------
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
.result{padding:1rem;background:#f8f9fa;border-radius:12px;font-size:1.25rem;color:#111;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎙️ Mongolian Fast-Whisper STT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>(v4.3.1 — Stable Custom Recorder)</p>", unsafe_allow_html=True)

# ---------- MODEL LOAD ----------
system, processor = platform.system().lower(), platform.processor().lower()
compute_type = "float32" if ("darwin" in system and "apple" in processor) else "int8"

@st.cache_resource(show_spinner=False)
def load_model():
    return WhisperModel("gana1215/MN_Whisper_Small_CT2", device="cpu", compute_type=compute_type)

with st.spinner("🔁 Loading Whisper model…"):
    model = load_model()
st.success("✅ Model loaded successfully!")

# ---------- AUDIO HELPERS ----------
def decode_any_audio(raw_bytes: bytes):
    """Decode any browser blob (webm, mp4, wav, ogg) into mono float32 @16kHz."""
    container = av.open(io.BytesIO(raw_bytes))
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if not stream:
        container.close()
        return np.zeros((0,), dtype=np.float32), 16000
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
    samples = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            frame = resampler.resample(frame)
            arr = frame.to_ndarray().astype(np.float32) / 32768.0
            if arr.ndim > 1:
                arr = np.mean(arr, axis=0)
            samples.append(arr)
    container.close()
    if not samples:
        return np.zeros((0,), dtype=np.float32), 16000
    return np.concatenate(samples), 16000

def write_temp_wav(data, sr):
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, data, sr)
    return path

def transcribe_wav(path):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(
            model.transcribe,
            path,
            language="mn",
            beam_size=1,
            vad_filter=False,
            without_timestamps=True,
            temperature=0.0
        )
        return fut.result(timeout=40)

# ---------- CUSTOM RECORDER (HTML + JS) ----------
rec = components.html(
    """
    <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
      <button id="recbtn" style="
          background:#0f4c81;color:white;border:none;border-radius:50%;
          width:110px;height:110px;font-size:1.1rem;box-shadow:0 4px 12px rgba(0,0,0,0.3);
          transition:all 0.25s ease;">🎙️ Start</button>
      <div id="status" style="color:#555;font-size:0.95rem;">Tap Start, speak, then tap Stop</div>
    </div>

    <script>
    let rec, chunks=[];
    const btn=document.getElementById("recbtn");
    const status=document.getElementById("status");

    async function startRec(){
      try{
        const stream=await navigator.mediaDevices.getUserMedia({audio:true});
        rec=new MediaRecorder(stream);
        chunks=[];
        rec.ondataavailable=e=>{if(e.data.size>0)chunks.push(e.data)};
        rec.onstop=async()=>{
          const blob=new Blob(chunks,{type:'audio/webm'});
          status.textContent="Processing...";
          const arrayBuffer=await blob.arrayBuffer();
          const bytes=Array.from(new Uint8Array(arrayBuffer));
          const json=JSON.stringify({audio:bytes,mime:blob.type||'audio/webm'});
          // Send dual messages (fix for Streamlit iframe timing)
          window.parent.postMessage({type:'streamlit:setComponentValue',value:json},'*');
          setTimeout(()=>window.parent.postMessage({type:'streamlit:setComponentValue',value:json},'*'),50);
          status.textContent="✅ Audio sent";
        };
        rec.start();
        btn.textContent="⏹ Stop";
        btn.style.background="#d32f2f";
        status.textContent="Recording...";
      }catch(err){
        status.textContent="❌ Mic permission denied";
      }
    }

    function stopRec(){
      if(rec && rec.state!=='inactive'){
        rec.stop();
        btn.textContent="🎙️ Start";
        btn.style.background="#0f4c81";
        status.textContent="Stopping...";
      }
    }

    btn.addEventListener('click',()=>{
      if(!rec || rec.state==='inactive') startRec(); else stopRec();
    });
    </script>
    """,
    height=230,
    scrolling=False,
    key="recorder_v431"
)

# ---------- PROCESS ----------
if isinstance(rec, str) and rec:
    try:
        msg = json.loads(rec)
        raw = bytes(msg["audio"])
        mime = msg.get("mime", "audio/webm")

        data, sr = decode_any_audio(raw)
        if len(data) < sr * 0.25:
            st.warning("⚠️ Too short or silent. Please try again.")
        else:
            st.info("⏳ Recognizing Mongolian speech…")
            t0 = time.time()
            wav = write_temp_wav(data, sr)
            segments, info = transcribe_wav(wav)
            os.unlink(wav)
            text = " ".join([s.text.strip() for s in segments if getattr(s, "text", "").strip()]) if segments else ""
            if text:
                st.success("✅ Recognition complete!")
                st.markdown(f"<div class='result'>{text}</div>", unsafe_allow_html=True)
                st.caption(f"⚡ {(time.time()-t0):.2f}s — MN_Whisper_Small_CT2 ({compute_type})")
                st.session_state["last_text"] = text
                st.session_state["last_audio_bytes"] = raw
                st.session_state["last_mime"] = mime
            else:
                st.warning("⚠️ No speech detected.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ---------- RETRY + LAST RESULT ----------
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔁 Retry last audio", use_container_width=True):
        if st.session_state.get("last_audio_bytes"):
            raw = st.session_state["last_audio_bytes"]
            mime = st.session_state.get("last_mime", "audio/webm")
            data, sr = decode_any_audio(raw)
            st.info("⏳ Re-transcribing last recording…")
            t0 = time.time()
            wav = write_temp_wav(data, sr)
            segments, info = transcribe_wav(wav)
            os.unlink(wav)
            text = " ".join([s.text.strip() for s in segments if getattr(s, "text", "").strip()]) if segments else ""
            if text:
                st.success("✅ Recognition complete!")
                st.markdown(f"<div class='result'>{text}</div>", unsafe_allow_html=True)
                st.caption(f"⚡ {(time.time()-t0):.2f}s — Retried")
                st.session_state["last_text"] = text
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

if st.session_state.get("last_text"):
    st.markdown("---")
    st.markdown(
        f"<p style='font-size:1.05rem;color:#444;'>🗣️ <b>Last recognized text:</b> {st.session_state['last_text']}</p>",
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#666;'>Developed by <b>Gankhuyag Mambaryenchin</b><br>"
    "Fine-tuned Whisper Model — Mongolian Fast-Whisper (v4.3.1 Stable Custom Recorder)</p>",
    unsafe_allow_html=True
)
