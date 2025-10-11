import os, pandas as pd

BASE_DIR = "Whisper_Data"

for file in os.listdir(BASE_DIR):
    path = os.path.join(BASE_DIR, file)
    if file.endswith(".csv"):
        try:
            df = pd.read_csv(path)
            for col in df.columns:
                df[col] = df[col].astype(str)
                for pattern in ["AI", "Ai", "ia", "Ia"]:
                    df[col] = df[col].str.replace(pattern, "Хиймэл оюун ухаан", regex=False)
            df.to_csv(path, index=False)
            print(f"✅ Fixed {file} ({len(df)} rows)")
        except Exception as e:
            print(f"⚠️ Skipped {file}: {e}")
    elif file.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            for pattern in ["AI", "Ai", "ia", "Ia"]:
                line = line.replace(pattern, "Хиймэл оюун ухаан")
            new_lines.append(line)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"✅ Fixed {file}")
