# ===============================================
# 🎙️ Mongolian Fast-Whisper STT (v2.0 — Final Cloud-Stable)
# ✅ Replaced streamlit_audio_recorder with st.audio_input
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

# ===============================================
# --- PAGE SETUP & STYLING ---
# ===============================================
st.set_page_config(
    page_title="🎙️ Mongolian Fast-Whisper STT",
    page_icon="🎧",
    layout="centered"
)

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
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    font-weight:800;
    text-align:center;
    margin-bottom:0.3rem;
}
.subtitle{text-align:center;font-size:1.1rem;color:#555;margin-bottom:1rem;font-style:italic;}
div.stButton>button:first-child{
    background:linear-gradient(90deg,#0f4c81,#1f8ac0);
    color:white;font-weight:bold;border-radius:12px;padding:0.6rem 1.2rem;border:none;
}
.stSuccess,.stInfo,.stWarning,.stError{border-radius:10px;}
div[data-testid="stAudioInput"] {
    text-align:center;
}
div[data-testid="stAudioInput"] label {
    display:block;
    font-weight:bold;
    margin-bottom:0.5rem;
    font-size:1.1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎙️ Mongolian Fast-Whisper STT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>(Anti-Hallucination Edition — Cloud-Stable)</p>", unsafe_allow_html=True)
st.caption("⚡ Fine-tuned Mongolian Whisper model with stable cloud inference")

# ===============================================
# --- MODEL LOADING (CACHED) ---
# ===============================================
system = platform.system().lower()
processor = platform.processor().lower()
if "darwin" in system and "apple" in processor:
    compute_type = "float32"
elif "darwin" in system:
    compute_type = "int8"
else:
    compute_type = "int8"  # Streamlit Cloud CPU

@st.cache_resource(show_spinner=False)
def load_model():
    repo_id = "gana1215/MN_Whisper_Small_CT2"
    model = WhisperModel(repo_id, device="cpu", compute_type=compute_type)
    return model

with st.spinner("🔁 Loading Whisper model..."):
    model = load_model()
st.success("✅ Model loaded successfully! Ready to transcribe your voice.")

# ===============================================
# --- AUDIO RECORDING SECTION (st.audio_input) ---
# ===============================================
st.subheader("🎤 Record your voice below")
st.write("Click the record button, speak in Mongolian, then click stop to transcribe:")

audio_file = st.audio_input("🎙️ Start recording")

if audio_file is not None:
    st.success(f"🎧 Recorded audio received — {audio_file.size} bytes")
    st.audio(audio_file, format="audio/wav")

    # --- STEP 1: Decode audio ---
    try:
        data, sr = sf.read(io.BytesIO(audio_file.read()))
        st.caption(f"📊 Audio decoded: {data.shape}, {sr} Hz")

        # --- STEP 2: Convert to 16kHz mono (for Whisper) ---
        if sr != 16000:
            data = scipy.signal.resample_poly(data, 16000, sr)
            sr = 16000
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # --- STEP 3: Save temporary file for Whisper ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp, data, sr)
            tmp_path = tmp.name

        # --- STEP 4: Transcribe ---
        st.info("⏳ Recognizing your Mongolian speech... please wait.")
        start = time.time()
        segments, info = model.transcribe(tmp_path, language="mn", beam_size=1)
        end = time.time()

        text = " ".join([s.text.strip() for s in segments if s.text.strip()])

        # --- STEP 5: Display result ---
        if text:
            st.success("✅ Recognition complete!")
            st.markdown("### 🗣️ Recognized Text:")
            st.markdown(
                f"<div style='padding:1rem;background:#f8f9fa;border-radius:12px;"
                f"font-size:1.3rem;color:#111;'>{text}</div>",
                unsafe_allow_html=True
            )
            st.caption(f"⚡ Processed in {end - start:.2f}s — Model: MN_Whisper_Small_CT2 ({compute_type})")
        else:
            st.warning("⚠️ No speech detected. Please try again closer to the mic.")

    except Exception as e:
        st.error(f"❌ Audio decoding error: {e}")

else:
    st.info("⏺️ Waiting for you to record...")

# ===============================================
# --- FOOTER ---
# ===============================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#666;'>Developed by <b>Gankhuyag Mambaryenchin</b><br>"
    "Fine-tuned Whisper Model — Mongolian Fast-Whisper (Anti-Hallucination Edition v2.0)</p>",
    unsafe_allow_html=True
)
