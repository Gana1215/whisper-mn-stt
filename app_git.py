# ===============================================
# 🎙️ Mongolian Fast-Whisper STT (v3.0.0 — Custom Recorder Core)
# ✅ Streamlit + Embedded JS Recorder (WAV) • iPhone Chrome/Safari compatible
# ✅ No audio widgets; instant in-memory transcription with faster-whisper
# ===============================================

import streamlit as st
from streamlit.components.v1 import html as st_html
import base64, json, io, os, time, platform
import numpy as np, soundfile as sf, scipy.signal
from faster_whisper import WhisperModel

# ---------- Environment (multi-threaded CT2 for speed) ----------
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("CT2_THREADS", "2")

# ---------- Page ----------
st.set_page_config(page_title="🎙️ Mongolian Fast-Whisper STT", page_icon="🎧", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+MN&display=swap');
html,body,[class*="css"]{font-family:'Noto Sans MN',sans-serif;background:radial-gradient(circle at top left,#f0f2f6,#dfe4ea);color:#222;}
h1{background:-webkit-linear-gradient(45deg,#0f4c81,#1f8ac0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800;text-align:center;margin-bottom:0.3rem;}
.subtitle{text-align:center;font-size:1.1rem;color:#555;margin-bottom:1rem;font-style:italic;}
.btn{display:inline-flex;align-items:center;gap:.5rem;border:none;border-radius:12px;padding:.6rem 1.2rem;font-weight:700;color:#fff;cursor:pointer;box-shadow:0 4px 10px rgba(0,0,0,.15)}
.btn-start{background:linear-gradient(90deg,#0f4c81,#1f8ac0)}
.btn-stop{background:linear-gradient(90deg,#d32f2f,#f44336)}
.small{color:#666;font-size:.9rem}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;}
.wave{height:42px;background:rgba(0,0,0,.05);border-radius:10px;position:relative;overflow:hidden}
.bar{position:absolute;bottom:0;width:3px;background:#0f4c81;left:0}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎙️ Mongolian Fast-Whisper STT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>(Custom Recorder • iPhone Chrome/Safari safe)</p>", unsafe_allow_html=True)

# ---------- Model ----------
system, processor = platform.system().lower(), platform.processor().lower()
compute_type = "int8" if ("darwin" not in system or "apple" not in processor) else "float32"

@st.cache_resource(show_spinner=False)
def load_model():
    # For near-instant CPU inference on Streamlit Cloud, Tiny is recommended.
    # If you prefer your Small model, swap names below (will be slower on Cloud).
    # return WhisperModel("gana1215/MN_Whisper_Small_CT2", device="cpu", compute_type=compute_type)
    return WhisperModel("gana1215/MN_Whisper_Tiny_CT2", device="cpu", compute_type=compute_type)

with st.spinner("🔁 Loading Whisper model…"):
    model = load_model()
st.success(f"✅ Model ready ({'Tiny' if 'Tiny' in str(model) else 'Small'}) — compute={compute_type}")

# ---------- Audio helpers ----------
def ensure_mono_16k(data: np.ndarray, sr: int):
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != 16000:
        data = scipy.signal.resample_poly(data, 16000, sr)
        sr = 16000
    return data.astype(np.float32), sr

def normalize_audio(x: np.ndarray):
    m = float(np.max(np.abs(x)) or 1e-9)
    if m < 0.005:  # almost silent
        return np.zeros_like(x)
    y = np.clip(x / m, -1.0, 1.0)
    return y

def transcribe(data: np.ndarray):
    # Fast path: no timestamps, no VAD, beam_size=1
    segments, info = model.transcribe(
        data,
        language="mn",
        beam_size=1,
        vad_filter=False,
        without_timestamps=True,
        temperature=0.0,
    )
    text = " ".join([s.text.strip() for s in segments if getattr(s, "text", "").strip()])
    return text

# ---------- Custom JS recorder (WAV in browser, base64 to Python) ----------
# Uses Web Audio API + ScriptProcessor to capture PCM, builds WAV in JS (mono)
# Works on iOS Safari/Chrome — no reliance on MediaRecorder codecs.
REC_HEIGHT = 240
recorder = st_html(f"""
<div style="display:flex;flex-direction:column;gap:10px;align-items:center">
  <div class="wave" id="wave"></div>
  <div style="display:flex;gap:10px;">
    <button id="startBtn" class="btn btn-start">🎙️ Start</button>
    <button id="stopBtn" class="btn btn-stop" disabled>🛑 Stop</button>
  </div>
  <div class="small">Tip: speak 2–5 seconds. Works on iPhone Chrome/Safari.</div>
</div>

<script>
(function() {{
  const wave = document.getElementById("wave");
  let audioCtx, processor, source, stream;
  let bufferL = [];  // mono
  let totalSamples = 0;
  let sampleRate = 16000; // target in WAV header; we'll resample in Python if device differs

  // Simple bars visualizer
  let barIdx = 0, bars = [];
  function visLevel(level) {{
    const h = Math.max(2, Math.min(40, Math.floor(level * 40)));
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.left = (barIdx*4%wave.clientWidth)+'px';
    bar.style.height = h+'px';
    wave.appendChild(bar);
    bars.push(bar);
    if (bars.length>200) {{
      const b = bars.shift(); if (b && b.parentNode) b.parentNode.removeChild(b);
    }}
    barIdx++;
  }}

  async function startRec() {{
    document.getElementById("startBtn").disabled = true;
    document.getElementById("stopBtn").disabled = false;
    bufferL = []; totalSamples = 0;

    // iOS requires user-gesture-created AudioContext
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({{sampleRate: 44100}});
    stream = await navigator.mediaDevices.getUserMedia({{audio: true}});
    source = audioCtx.createMediaStreamSource(stream);

    const bufferSize = 4096;
    processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
    processor.onaudioprocess = function(e) {{
      const input = e.inputBuffer.getChannelData(0);
      // downmix to mono (already mono), store float32 chunk
      bufferL.push(new Float32Array(input));
      totalSamples += input.length;

      // simple level for visual
      let max = 0.0;
      for (let i=0;i<input.length;i++) {{
        const v = Math.abs(input[i]);
        if (v>max) max = v;
      }}
      visLevel(max);
    }};
    source.connect(processor);
    processor.connect(audioCtx.destination);
  }}

  function mergeFloat32(buffers, total) {{
    const out = new Float32Array(total);
    let off = 0;
    for (let i=0;i<buffers.length;i++) {{ out.set(buffers[i], off); off += buffers[i].length; }}
    return out;
  }}

  function floatTo16BitPCM(float32) {{
    const len = float32.length;
    const out = new Int16Array(len);
    for (let i=0;i<len;i++) {{
      let s = Math.max(-1, Math.min(1, float32[i]));
      out[i] = (s < 0 ? s * 0x8000 : s * 0x7FFF);
    }}
    return out;
  }}

  function writeWav(samples, sampleRate) {{
    const byteRate = sampleRate * 2; // mono 16-bit
    const blockAlign = 2;
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    function writeStr(offset, s) {{ for (let i=0;i<s.length;i++) view.setUint8(offset+i, s.charCodeAt(i)); }}
    let p = 0;
    writeStr(p, "RIFF"); p += 4;
    view.setUint32(p, 36 + samples.length*2, true); p += 4;
    writeStr(p, "WAVE"); p += 4;
    writeStr(p, "fmt "); p += 4;
    view.setUint32(p, 16, true); p += 4; // PCM header size
    view.setUint16(p, 1, true); p += 2;  // PCM
    view.setUint16(p, 1, true); p += 2;  // mono
    view.setUint32(p, sampleRate, true); p += 4;
    view.setUint32(p, byteRate, true); p += 4;
    view.setUint16(p, blockAlign, true); p += 2;
    view.setUint16(p, 16, true); p += 2; // 16-bit
    writeStr(p, "data"); p += 4;
    view.setUint32(p, samples.length*2, true); p += 4;

    // PCM samples
    const pcm = floatTo16BitPCM(samples);
    for (let i=0, off=44; i<pcm.length; i++, off+=2) view.setInt16(off, pcm[i], true);
    return new Blob([view], {{ type: "audio/wav" }});
  }}

  async function stopRec() {{
    document.getElementById("stopBtn").disabled = true;

    if (processor) {{ processor.disconnect(); processor.onaudioprocess = null; }}
    if (source) source.disconnect();
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (audioCtx && audioCtx.state !== "closed") await audioCtx.close();

    const merged = mergeFloat32(bufferL, totalSamples);
    // NOTE: We're keeping the device sample rate (usually 44100). Python will resample to 16k.
    const wavBlob = writeWav(merged, 44100);

    const reader = new FileReader();
    reader.onloadend = function() {{
      const b64 = reader.result.split(",")[1];  // strip "data:audio/wav;base64,"
      const payload = {{
        kind: "audio_wav_base64",
        mime: "audio/wav",
        b64: b64,
        samples: merged.length
      }};
      const out = {{
        isStreamlitMessage: true,
        type: "streamlit:setComponentValue",
        value: JSON.stringify(payload)
      }};
      window.parent.postMessage(out, "*");
    }};
    reader.readAsDataURL(wavBlob);

    // Re-enable start after sending
    document.getElementById("startBtn").disabled = false;
  }}

  document.getElementById("startBtn").onclick = (e) => {{ e.preventDefault(); startRec(); }};
  document.getElementById("stopBtn").onclick  = (e) => {{ e.preventDefault(); stopRec(); }};
}})();
</script>
""", height=REC_HEIGHT)

# ---------- Receive value from the HTML component ----------
raw_val = recorder  # components.html returns the last value set via postMessage

# ---------- State ----------
if "last_text" not in st.session_state: st.session_state["last_text"] = ""
if "last_audio_bytes" not in st.session_state: st.session_state["last_audio_bytes"] = None

def handle_audio_bytes(wav_bytes: bytes):
    # Decode WAV → float32, resample → 16k, normalize → transcribe
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception as e:
        st.error(f"Decode failed: {e}")
        return

    data, sr = ensure_mono_16k(data, sr)
    data = normalize_audio(data)
    if np.abs(data).max() < 0.02 or len(data) < sr * 0.25:
        st.warning("⚠️ Recording too short or too quiet.")
        return

    st.info("⏳ Recognizing Mongolian speech…")
    t0 = time.time()
    try:
        text = transcribe(data)
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return
    dt = time.time() - t0

    st.session_state["last_text"] = text
    st.session_state["last_audio_bytes"] = wav_bytes

    if text.strip():
        st.success("✅ Recognition complete!")
        st.markdown(
            f"<div style='padding:1rem;background:#f8f9fa;border-radius:12px;font-size:1.3rem;color:#111;'>{text}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"⚡ {dt:.2f}s — Faster-Whisper ({compute_type})")
    else:
        st.warning("⚠️ No speech detected.")

# ---------- If we received audio from JS, process it ----------
if raw_val:
    try:
        payload = json.loads(raw_val)
        if isinstance(payload, str):  # defensive: sometimes double-encoded
            payload = json.loads(payload)
    except Exception:
        payload = None

    if payload and payload.get("kind") == "audio_wav_base64":
        b64 = payload.get("b64")
        try:
            wav_bytes = base64.b64decode(b64)
        except Exception as e:
            st.error(f"Base64 decode failed: {e}")
            wav_bytes = None
        if wav_bytes:
            handle_audio_bytes(wav_bytes)

# ---------- Retry button ----------
if st.button("🔁 Retry last audio"):
    if st.session_state.get("last_audio_bytes"):
        handle_audio_bytes(st.session_state["last_audio_bytes"])
    else:
        st.info("No previous audio to retry yet.")

# ---------- Last text ----------
if st.session_state["last_text"]:
    st.markdown("---")
    st.markdown(
        f"<p style='font-size:1.05rem;color:#444;'>🗣️ <b>Last recognized text:</b> {st.session_state['last_text']}</p>",
        unsafe_allow_html=True,
    )

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#666;'>Developed by <b>Gankhuyag Mambaryenchin</b><br>"
    "Mongolian Fast-Whisper — Custom Recorder Core v3.0.0</p>",
    unsafe_allow_html=True,
)
