# ===============================================
# 🎙️ Whisper Mongolian STT — Streamlit Cloud + Dropbox
# ===============================================
import os
import io
import torch
import torchaudio
import tempfile
import streamlit as st
import dropbox
from st_audiorec import st_audiorec
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration

# --- Streamlit Config ---
st.set_page_config(page_title="Whisper Mongolian STT", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+MN&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans MN', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("🎙️ Mongolian Speech-to-Text (Whisper Anti-Hallucination Edition)")

# ===============================================
# 🔐 Dropbox Setup
# ===============================================
DROPBOX_TOKEN = st.secrets["sl.u.AGBsMFPVYB-aaGeaGaYpAq9CX0AVFSZOgR8TN9YIUS97gDPNe8OGQK4l2HDuZUJ5n5gpgxb4Xk7IFelDotsdyUt_atKF2igacEgVDF4uLUiJa7XbAHTDXX8YVWMg-9dNA3vaGDvHhYwCezlqgX4Ltkqv43LJAqG7-jrUcgQV-6t-YNopFcV_R0-9wZakznCnnwZHlYEPTZqki9MWfO-BkoyBAWD6Cn0I3Xllx265J4gbH9H8qpTJSrxj7fOHJP0dJOJyDCHg3sMkHuCqzn0gQ8B9ezsvee_IkWbhJb9BMdZxxwykXUXFdYmwVBdAPToLym7BTx5jJwddIZBsWmLGPdffk1thiVGWU0IJst5L8tjjRF584g96_B3p6vrSHLWOkNl7o3SDP06L4isz0vvvQp36WWzBCJLFbx6b6nBD5IqaHNgH_jllYaC2WZ9nHQsk23kFYCIpwjQoWURiKaYhj99-yuYJrn7iZPvuIAOkobOINqMMO_4iI-JbXmUbJbIGyBMNYTaWU8FClkazWaTdFRmXZaTckmRAJKWGUcflrJyk-AkZ5IsgO0NbEHWIFON9LcNJXijAFJ3Kv_ImuM7WI9oSaGKlqRBdB2EWn_MY3YQCVb4RbXpLntWne_3a1b6KIbpFkKg-8QDmOBni9MlDSDN11nVoAY6Ba6NBDKZ2_6KKfJTmg3nE1cLJrwSuA4QUDOKMyI4T24IoVCLE82EniUv9fUxOPWZlkUfCv3AiusxkbTv3ouTADObrdDc7_E4uUDrrs3Fyg3vBBRsVhP11YxM664dmJbCV3QFt0DaOFARy70I67kWywiVhNl-lVJIXtltfX13pwWgqPr2dhhmdYbfUc08Q51n1ApQPWkuw2-7ReXyEY5pCldt9jAOeexvt8uARxxXjwLt9ZUIJhxfr6ZNDxvq4gz7xc2xw46WYOCNAAH0g5QhaOqlk7jhPd9HrRjM4JghJluIOQ2qCxLrYbb0Kd3J9-3p7hUit322Fm8HxjhgFVJzUbSxqOYR7O1dJZ42EWpCH_BSpVtRWe-hJfWUEEfaa0mlY1sPbheZkp-CPL4lmLFPe1az9TK-dE5R-YbOAPGSKLfpuRubQQ3rdpar6HQV0DHLk24ITPhlhdzdvo84CoB_WeYJY4NKjVs8LGefJpCxs_gI5qgvACug1k4_74c_g_kb2j9mS8NCBFH58GZmAqliSnwooBkSZtm_-pd8voVz-boRGxYWS4luv-kKVysDXlYvN_jlVJTrXIRQeNDwR2v2iiND2A2qQ2BZyqNSi21tDxf4tQTnKRx4Iwk-NlKccZSXg-AyGs1KK4AUuY-G2UR5wEQA93uCv2QO6ljmby4vrOPFqiaFpl-f1wMjd7Wp1cdubd8LpO3V8j-mHnHCh7qMnJ0nrgtITGw0RC8MuMs_UGbiyHl1On3FmKuAUUOrEKhysY4colICbfhE_yQ"]
MODEL_FOLDER = "/models/checkpoint-3500"
LOCAL_MODEL_DIR = "./models/checkpoint-3500"

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

@st.cache_resource(show_spinner=True)
def download_model_from_dropbox():
    """Download Whisper model files from Dropbox (only if not cached)."""
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)
    st.info("🔄 Checking and downloading model from Dropbox...")

    for entry in dbx.files_list_folder(MODEL_FOLDER).entries:
        if isinstance(entry, dropbox.files.FileMetadata):
            file_name = entry.name
            local_path = os.path.join(LOCAL_MODEL_DIR, file_name)
            if not os.path.exists(local_path):
                _, res = dbx.files_download(path=entry.path_display)
                with open(local_path, "wb") as f:
                    f.write(res.content)
    return LOCAL_MODEL_DIR

MODEL_PATH = download_model_from_dropbox()
st.success("✅ Model downloaded successfully!")

# ===============================================
# 🎧 Load Whisper Model
# ===============================================
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

forced_ids = processor.get_decoder_prompt_ids(language="mn", task="transcribe")
generate_kwargs = {
    "forced_decoder_ids": forced_ids,
    "do_sample": False,
    "temperature": 0.0,
    "num_beams": 5,
    "max_new_tokens": 32,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.05,
    "length_penalty": 0.1,
}

asr = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=15,
    stride_length_s=(2, 2),
    generate_kwargs=generate_kwargs,
    device=0 if torch.cuda.is_available() else -1,
)
st.success("✅ Whisper model loaded and ready!")

# ===============================================
# 🎚️ Silence Trimming Helper
# ===============================================
def trim_silence(audio, sr=16000, thresh=0.005):
    frame_len = int(sr * 0.03)
    frames = audio.unfold(0, frame_len, frame_len)
    energy = (frames ** 2).mean(dim=1)
    mask = energy > thresh

    if not mask.any():
        return audio
    mask_int = mask.int()
    start = (mask_int.argmax().item()) * frame_len
    end = (len(mask_int) - torch.flip(mask_int, [0]).argmax().item()) * frame_len
    return audio[start:end].contiguous()

# ===============================================
# 🎤 Audio Recorder + Transcription
# ===============================================
st.subheader("🎤 Record your voice below:")
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_audio_data)
        temp_path = tmp.name

    wav, sr = torchaudio.load(temp_path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = trim_silence(wav.squeeze(0))
    torchaudio.save(temp_path, wav.unsqueeze(0), 16000)

    st.markdown("---")
    st.info("⏳ Recognizing your voice... please wait")

    result = asr(temp_path)
    st.success("✅ Recognition complete!")
    st.markdown("### 🗣️ Recognized Text:")
    st.markdown(f"<h2>{result['text']}</h2>", unsafe_allow_html=True)
