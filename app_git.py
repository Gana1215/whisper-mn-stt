# ===============================================
# 🎙️ Mongolian Fast-Whisper STT (v4.0 — Custom Component Recorder)
# ✅ Works on iPhone Safari/Chrome, Android, Mac, Windows
# ✅ No st-audiorec / st.audio_input
# ✅ Proper Streamlit component bridge (postMessage -> setComponentValue)
# ===============================================

import os, io, base64, tempfile, time, platform, concurrent.futures
import numpy as np
import soundfile as sf
import av
import streamlit as st
import streamlit.components.v1 as components
from faster_whisper import WhisperModel

# ---------------------- Page ----------------------
st.set_page_config(page_title="🎙️ Mongolian Fast-Whisper STT", page_icon="🎧", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+MN&display=swap');
html,body,[class*="css"]{font-family:'Noto Sans MN',sans-serif;background:radial-gradient(circle at top left,#f0f2f6,#dfe4ea);color:#222;}
h1{background:-webkit-linear-gradient(45deg,#0f4c81,#1f8ac0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800;text-align:center;margin-bottom:0.4rem;}
.subtitle{text-align:center;font-size:1.05rem;color:#555;margin-bottom:1rem;font-style:italic;}
#recwrap{display:flex;flex-direction:column;align-items:center;gap:.6rem}
.recbtn{background:#0f4c81;color:#fff;border:none;border-radius:50%;width:104px;height:104px;font-size:1.05rem;box-shadow:0 4px 10px rgba(0,0,0,.25);transition:all .25s ease}
.recbtn.recording{background:#d32f2f;box-shadow:0 0 22px #ff4d4d}
.small{color:#666;font-size:.95rem}
.result{padding:1rem;background:#f8f9fa;border-radius:12px;font-size:1.25rem;color:#111;}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1>🎙️ Mongolian Fast-Whisper STT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>(Custom Component Recorder — Universal Browser Safe)</p>", unsafe_allow_html=True)

# ---------------------- Model (kept exactly as your working setup) ----------------------
system, processor = platform.system().lower(), platform.processor().lower()
compute_type = "float32" if ("darwin" in system and "apple" in processor) else "int8"

@st.cache_resource(show_spinner=True)
def load_model():
    return WhisperModel("gana1215/MN_Whisper_Small_CT2", device="cpu", compute_type=compute_type)

with st.spinner("🔁 Loading Whisper model…"):
    model = load_model()
st.success("✅ Model loaded successfully!")

# ---------------------- Audio helpers ----------------------
def decode_webm_to_float32_mono_16k(webm_bytes: bytes):
    """Decode WebM/Opus -> float32 mono 16k using PyAV (no external ffmpeg binary)."""
    container = av.open(io.BytesIO(webm_bytes))
    astream = next(s for s in container.streams if s.type == "audio")
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    for pkt in container.demux(astream):
        for frame in pkt.decode():
            f = resampler.resample(frame)
            arr = f.to_ndarray()
            if arr.ndim > 1:
                arr = np.mean(arr, axis=0)
            chunks.append(arr.astype(np.float32) / 32768.0)
    container.close()
    audio = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)
    return audio, 16000

def write_temp_wav(data: np.ndarray, sr: int):
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, data, sr)
    return path

def transcribe_path(path: str):
    # Fast and robust defaults
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(
            model.transcribe,
            path,
            language="mn",
            beam_size=1,
            vad_filter=False,            # lower latency
            without_timestamps=True,
            temperature=0.0
        )
        return fut.result(timeout=40)

# ---------------------- Recorder component ----------------------
# This component returns `{"b64": "...", "mime": "audio/webm"}` after STOP
recorder_value = components.html(
    """
    <div id="recwrap">
      <button id="recbtn" class="recbtn">🎙️ Start</button>
      <div id="hint" class="small">Tap Start, speak, then tap Stop</div>
      <div id="status" class="small">Ready</div>
    </div>

    <script>
    const btn = document.getElementById("recbtn");
    const status = document.getElementById("status");
    let mediaRecorder = null;
    let chunks = [];

    async function startRec(){
      try{
        const stream = await navigator.mediaDevices.getUserMedia({audio:true});
        chunks = [];
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (e) => { chunks.push(e.data); };
        mediaRecorder.onstop = async () => {
          try{
            const blob = new Blob(chunks, {type: 'audio/webm'});
            status.innerText = "Processing…";
            const buf = await blob.arrayBuffer();
            const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)));

            // ✅ Send value back to Streamlit (OFFICIAL BRIDGE)
            window.parent.postMessage({
              type: 'streamlit:setComponentValue',
              value: { b64: b64, mime: blob.type }
            }, '*');

            status.innerText = "✅ Audio sent";
          }catch(err){
            status.innerText = "❌ Failed to process audio";
          }
          btn.classList.remove('recording');
          btn.innerText = "🎙️ Start";
        };

        mediaRecorder.start();
        btn.classList.add('recording');
        btn.innerText = "⏹ Stop";
        status.innerText = "Recording…";
      }catch(err){
        status.innerText = "❌ Mic permission denied";
      }
    }

    function stopRec(){
      if (mediaRecorder && mediaRecorder.state !== "inactive"){
        mediaRecorder.stop();
        status.innerText = "Stopping…";
      }
    }

    btn.addEventListener('click', ()=>{
      if (!mediaRecorder || mediaRecorder.state === "inactive") startRec();
      else stopRec();
    });
    </script>
    """,
    height=240,
    scrolling=False,
)

# ---------------------- Process result ----------------------
last_text = st.session_state.get("last_text", "")
last_audio_b64 = st.session_state.get("last_audio_b64", None)
last_audio_mime = st.session_state.get("last_audio_mime", "audio/webm")

if recorder_value:
    # recorder_value is a JSON-like dict set by streamlit:setComponentValue
    try:
        b64 = recorder_value.get("b64") if isinstance(recorder_value, dict) else None
        mime = recorder_value.get("mime") if isinstance(recorder_value, dict) else "audio/webm"
    except Exception:
        b64, mime = None, "audio/webm"

    if b64:
        st.info("⏳ Transcribing your voice…")
        try:
            raw = base64.b64decode(b64)
            # Decode WebM/Opus -> float32 mono 16k
            audio, sr = decode_webm_to_float32_mono_16k(raw)
            if audio.size < sr * 0.25:
                st.warning("⚠️ Recording too short. Please try again closer to the mic.")
            else:
                wav = write_temp_wav(audio, sr)
                t0 = time.time()
                segments, info = transcribe_path(wav)
                try:
                    os.unlink(wav)
                except Exception:
                    pass

                text = " ".join([s.text.strip() for s in segments if getattr(s, "text", "").strip()]) if segments else ""
                st.session_state["last_text"] = text
                st.session_state["last_audio_b64"] = b64
                st.session_state["last_audio_mime"] = mime

                if text:
                    st.success("✅ Recognition complete!")
                    st.markdown(f"<div class='result'>{text}</div>", unsafe_allow_html=True)
                    st.caption(f"⚡ {(time.time()-t0):.2f}s — MN_Whisper_Small_CT2 ({compute_type})")
                else:
                    st.warning("⚠️ No speech detected.")
        except Exception as e:
            st.error(f"❌ Error while processing: {e}")

# ---------------------- Retry last audio ----------------------
st.markdown(" ")
col1, col2 = st.columns([1,1])
with col1:
    if st.button("🔁 Retry last audio", use_container_width=True):
        if last_audio_b64:
            st.info("⏳ Transcribing previous recording…")
            try:
                raw = base64.b64decode(last_audio_b64)
                audio, sr = decode_webm_to_float32_mono_16k(raw)
                wav = write_temp_wav(audio, sr)
                segments, info = transcribe_path(wav)
                try:
                    os.unlink(wav)
                except Exception:
                    pass
                text = " ".join([s.text.strip() for s in segments if getattr(s, "text", "").strip()]) if segments else ""
                st.session_state["last_text"] = text
                if text:
                    st.success("✅ Recognition complete!")
                    st.markdown(f"<div class='result'>{text}</div>", unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No speech detected.")
            except Exception as e:
                st.error(f"❌ Error while retrying: {e}")
        else:
            st.info("No previous audio to retry yet.")

with col2:
    if last_audio_b64:
        # Offer download of last recording as WebM
        st.download_button(
            "⬇️ Download last recording",
            data=base64.b64decode(last_audio_b64),
            file_name="recording.webm",
            mime=last_audio_mime,
            use_container_width=True
        )
    else:
        st.button("⬇️ Download last recording", disabled=True, use_container_width=True)

# ---------------------- Last text (single render) ----------------------
if st.session_state.get("last_text"):
    st.markdown("---")
    st.markdown(
        f"<p class='small'>🗣️ <b>Last recognized text:</b> {st.session_state['last_text']}</p>",
        unsafe_allow_html=True
    )

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#666;'>Developed by <b>Gankhuyag Mambaryenchin</b><br>"
    "Fine-tuned Whisper Model — Mongolian Fast-Whisper (v4.0 Custom Component Recorder)</p>",
    unsafe_allow_html=True,
)
