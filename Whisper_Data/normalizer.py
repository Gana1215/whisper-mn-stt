# ===============================================
# 🧹 Whisper Dataset Universal Normalizer (Smart ID Handling)
# ===============================================
import os, pandas as pd, re

BASE_DIR = "Whisper_Data"
AI_REGEX = re.compile(r'\b[aA][\.\s]*[iI][\.\s]*\b', re.UNICODE)

def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    text = re.sub(AI_REGEX, "Хиймэл оюун ухаан ", text)
    return text.strip()

for file in os.listdir(BASE_DIR):
    if not file.endswith(".csv"):
        continue

    path = os.path.join(BASE_DIR, file)
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
        df.columns = [c.strip().lower() for c in df.columns]

        # Detect text column
        text_col = next((c for c in df.columns if c in ["sentence", "text", "expression", "phrase", "utterance"]), None)
        if text_col is None:
            text_col = df.columns[-1]

        # Clean text and rename
        df["text"] = df[text_col].apply(clean_text)

        # Handle ID (reuse or recreate)
        if "id" not in df.columns:
            df.insert(0, "id", range(1, len(df) + 1))
        else:
            df["id"] = range(1, len(df) + 1)

        # Keep only id,text
        df = df[["id", "text"]]

        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"✅ Normalized {file} ({len(df)} rows)")

    except Exception as e:
        print(f"⚠️ Skipped {file}: {e}")

print("\n🎯 All CSVs unified to [id,text] — existing IDs handled safely.")
