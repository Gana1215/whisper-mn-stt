import os
from datetime import datetime

#BASE_DIR = "/content/Gankhuyag_Whisper_Data"  # For Colab
BASE_DIR = "Whisper_Data"      # For local machine

os.makedirs(BASE_DIR, exist_ok=True)

word_list_content = """данс
мөнгө
гүйлгээ
... (2000 words total)
"""

expressions_content = """id,expression
0001,данс шалгах
0002,мөнгө авах
...
"""

sentences_content = """id,sentence
0001,Би дансаа шалгамаар байна.
0002,Эгчийн дансанд мөнгө орлоо.
...
"""

numbers_currency_content = """нэг төгрөг
хоёр төгрөг
гурван төгрөг
нэг мянган төгрөг
хоёр сая төгрөг
...
"""

with open(f"{BASE_DIR}/word_list.txt", "w", encoding="utf-8") as f:
    f.write(word_list_content)
with open(f"{BASE_DIR}/expressions.csv", "w", encoding="utf-8") as f:
    f.write(expressions_content)
with open(f"{BASE_DIR}/sentences.csv", "w", encoding="utf-8") as f:
    f.write(sentences_content)
with open(f"{BASE_DIR}/numbers_currency.txt", "w", encoding="utf-8") as f:
    f.write(numbers_currency_content)

print("✅ All dataset files created in:", BASE_DIR)
