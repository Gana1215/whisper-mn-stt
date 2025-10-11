# ===============================================
# 🎙️ Whisper Mongolian STT — Streamlit App
# ===============================================
import os
import torch
import torchaudio
import soundfile as sf
import tempfile
import streamlit as st
from st_audiorec import st_audiorec
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration

# --- App Title ---
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

# --- Load Model & Processor ---

MODEL_PATH = "./models/checkpoint-3500"   # 🔁 or your Drive path after fine-tuning
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

# --- Configure Anti-Hallucination Decoding ---
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

# --- Build ASR Pipeline ---
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
st.success("✅ Model loaded and ready!")

# --- VAD helper to trim silence ---
def trim_silence(audio, sr=16000, thresh=0.005):
    frame_len = int(sr * 0.03)
    frames = audio.unfold(0, frame_len, frame_len)
    energy = (frames ** 2).mean(dim=1)
    mask = energy > thresh

    if not mask.any():
        return audio

    # 🔧 fix: cast mask to int for argmax
    mask_int = mask.int()
    start = (mask_int.argmax().item()) * frame_len
    end = (len(mask_int) - torch.flip(mask_int, [0]).argmax().item()) * frame_len
    return audio[start:end].contiguous()

# --- Audio Recorder ---
st.subheader("🎤 Record your voice below:")
wav_audio_data = st_audiorec()

# --- Transcription ---
if wav_audio_data is not None:
    # Save temporary WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_audio_data)
        temp_path = tmp.name

    # Load + preprocess
    wav, sr = torchaudio.load(temp_path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = trim_silence(wav.squeeze(0))
    torchaudio.save(temp_path, wav.unsqueeze(0), 16000)

    # Clear previous output
    st.markdown("---")
    st.info("⏳ Recognizing your voice... please wait")

    # Recognize
    result = asr(temp_path)
    st.success("✅ Recognition complete!")
    st.markdown("### 🗣️ Recognized Text:")
    st.markdown(f"<h2>{result['text']}</h2>", unsafe_allow_html=True)
