# ===============================================
# Gankhuyag Whisper Dataset: Full Generator (≈20k)
# Mongolian (bank + daily + tech + numbers/currency)
# ===============================================
import os, csv, random, math, itertools, hashlib
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# ------------ CONFIG ------------
# Change this if running locally (e.g., "Gankhuyag_Whisper_Data")
BASE_DIR = "Whisper_Data"
TARGET_SENTENCES = 20000         # ≈ 20k meaningful sentences
TARGET_EXPRESSIONS = 8000        # 2–3 word expressions
SEED = 42                        # reproducibility

random.seed(SEED)
os.makedirs(BASE_DIR, exist_ok=True)

# ------------ CORE LEXICON (seed sets) ------------
# Bank & finance nouns
BANK = [
  "данс","гүйлгээ","мөнгө","төлбөр","арилжаа","худалдаа","валют","ханш","ашиг","алдагдал",
  "орлого","зарлага","баланс","зээл","карт","хадгаламж","авлага","өр","нотлох баримт",
  "баримт","квитанц","терминал","ATM","POS төхөөрөмж","карт уншигч","салбар","банк",
  "дансны дугаар","дансны нэр","интернэт банк","цахим банк","шилжүүлэг"
]

# Tech & digital
TECH = [
  "бар код","QR код","скан хийх","сканнер","хуулбарлах","копи хийх","принтер","веб хуудас",
  "веб сайт","апп","програм","цахим баримт","цахим үйлчилгээ","нууц үг","пин код",
  "баталгаажуулах код","OTP код","имэйл","цахим шуудан","wifi","bluetooth","notification",
  "update","install","download","upload","нэвтрэх","гарах","хандалт","файл","файл хуулбар",
  "файл илгээх","файл устгах","зураг","видео","код","алгоритм","мэдээллийн технологи",
  "өгөгдөл","өгөгдлийн сан","хиймэл оюун","хиймэл оюун ухаан","AI","IA","машин сургалт",
  "нейрон сүлжээ","робот","робот систем"
]

# Daily life nouns
DAILY = [
  "дэлгүүр","супермаркет","зах","ресторан","кафе","гэр","орон сууц","байшин","өрөө","гудамж","талбай",
  "машин","автобус","галт тэрэг","онгоц","шатахуун","зогсоол","эмнэлэг","эмч","эм","өвчин","эмчилгээ",
  "кофе","цай","ус","шар айраг","талх","гурил","будаа","мах","ногоо","чихэр","амттан","сүү","өндөг","салат",
  "муур","нохой","шувуу","загас","хүүхэд","багш","оюутан","ажилтан","менежер","удирдлага","найз","ах","эгч"
]

# Weather & nature
NATURE = [
  "цаг агаар","нар","бороо","цас","салхи","салхитай","бороотой","бүүдгэр","цэлмэг","дулаан","хүйтэн","халуун","сэрүүн",
  "уул","гол","нуур","мод","цэцэг","навч","шороо","зам"
]

# Time words
TIME = [
  "өнөөдөр","маргааш","өчигдөр","өглөө","өдөр","орой","шөнө","7 цагт","8 цагт","өнгөрсөн долоо хоногт","дараа сарын эхэнд"
]

# People / roles
PEOPLE = [
  "би","чи","тэр","ах","эгч","аав","ээж","найз","үйлчлүүлэгч","хэрэглэгч","хамт олон","менежер","кастомер","хүүхэд"
]

# Verbs (bank + general)
VERBS = [
  "шалгах","төлөх","авах","өгөх","нээх","хаах","шилжүүлэх","баталгаажуулах","илгээх","хүлээн авах",
  "асуух","хариулах","татах","оруулж өгөх","сонгох","засах","устгах","хайх","дансаа шалгах",
  "гүйлгээ хийх","валют солих","зээл авах","карт ашиглах","код бичих","файл илгээх","скан хийх",
  "хуулбарлах","хэвлэх","арилжаа хийх","худалдаж авах","худалдах","судлах","бичих","унших","уух","идэх",
  "бодолхийлэх","турших","зөвлөх","тоглох","аялах","амрах","ярилцах","залгах","очих","явах","ирэх"
]

EMOTION = ["баяртай","гунигтай","ууртай","санаа зовсон","итгэлтэй","найдвартай","тайван","яаралтай"]

# Currency units
CURRENCIES = ["төгрөг","доллар","евро","юань","рубль"]

# ------------ Numbers → Mongolian words (basic) ------------
ONES = ["нэг","хоёр","гурван","дөрвөн","таван","зургаан","долоон","найман","есөн"]
ONES_PLAIN = ["нэг","хоёр","гурав","дөрөв","тав","зургаа","долоо","найм","ес"]
TENS = ["арав","хорин","гучин","дөчин","тавин","жар","дал","ная","ер"]
HUNDRED = "зуу"
THOUSAND = "мянга"
THOUSAND_MOD = "мянган"
MILLION = "сая"
BILLION = "тэрбум"

def two_digit(n):
    if n < 10: return ONES_PLAIN[n-1]
    if n == 10: return "арав"
    if 11 <= n <= 19:
        return "арван " + ONES_PLAIN[n-10-1]
    t = n // 10; u = n % 10
    base = TENS[t-1]
    if u == 0: return base
    return base + " " + ONES_PLAIN[u-1]

def three_digit(n):
    if n < 100: return two_digit(n)
    h = n // 100; rest = n % 100
    hword = (ONES[h-1] + " " + HUNDRED)
    if rest == 0: return hword
    return hword + " " + two_digit(rest)

def number_words(n):
    if n < 1000: return three_digit(n)
    if n < 1000000:
        k = n // 1000; rest = n % 1000
        kth = number_words(k) + " " + THOUSAND
        if rest == 0: return kth
        return kth + " " + three_digit(rest)
    if n < 1000000000:
        m = n // 1000000; rest = n % 1000000
        mth = number_words(m) + " " + MILLION
        if rest == 0: return mth
        # ensure spacing
        part = (number_words(rest))
        return mth + " " + part
    b = n // 1000000000; rest = n % 1000000000
    bth = number_words(b) + " " + BILLION
    if rest == 0: return bth
    return bth + " " + number_words(rest)

def thousand_modified(n):
    # e.g., "нэг мянган төгрөг" (modifier form)
    if n == 1: return "нэг " + THOUSAND_MOD
    # for 2..9, use ONES + мянган
    if 2 <= n <= 9: return ONES[n-1] + " " + THOUSAND_MOD
    # default fallback
    return number_words(n) + " " + THOUSAND

# ------------ Build word_list (≈2000+) ------------
BASE_WORDS = sorted(set(BANK + TECH + DAILY + NATURE + TIME + PEOPLE + VERBS + EMOTION))
word_list = list(BASE_WORDS)

# Add numeric/currency tokens
for n in [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,500,700,900]:
    for cur in CURRENCIES:
        word_list.append(f"{number_words(n)} {cur}")
for n in range(1,10):
    word_list.append(thousand_modified(n))  # "нэг мянган", "хоёр мянган", ...

# Add extra tech/AI slang variants
word_list += ["QR гүйлгээ","бар код уншуулах","цахим нэвтрэлт","дансны үлдэгдэл","харагдах байдал","форм","талбар"]

# Dedup + sort
word_list = sorted(set(w.strip() for w in word_list if w.strip()))

# Write word_list.txt
with open(os.path.join(BASE_DIR,"word_list.txt"),"w",encoding="utf-8") as f:
    f.write("\n".join(word_list))

# ------------ Build expressions (2–3 word phrases) ------------
def pairs(a, b, verb_last=True):
    out = []
    for x in a:
        for y in b:
            if verb_last: out.append(f"{x} {y}")
            else: out.append(f"{y} {x}")
    return out

EXP_BANK = []
EXP_BANK += [f"{n} {v}" for n in ["данс","гүйлгээ","төлбөр","валют","зээл","карт"] for v in ["шалгах","төлөх","авах","нээх","хаах","шилжүүлэх","баталгаажуулах"]]
EXP_TECH = [
  "бар код уншуулах","QR код скан хийх","файл илгээх","файл хуулбарлах","файл устгах","принтер ашиглах",
  "цахим баримт илгээх","нууц үг сэргээх","пин код оруулах","баталгаажуулах код илгээх","апп нээх","апп хаах",
  "веб хуудас үзэх","имэйл илгээх","код бичих","өгөгдөл боловсруулах","машин сургалт хийх","нейрон сүлжээ ашиглах"
]
EXP_DAILY = [
  "кофе уух","цай уух","дэлгүүр явах","ресторан орох","кафе суух","ах руу залгах","гэрт очих","ажилдаа явах",
  "ном унших","кино үзэх","бороо орох","цас орох","нар гарах","салхи үлээх"
]

# Combine & trim to target
expressions = sorted(set(EXP_BANK + EXP_TECH + EXP_DAILY))
# If still short, auto-generate noun+verb pairs
if len(expressions) < TARGET_EXPRESSIONS:
    nouns = BANK + DAILY + TECH
    verbs = [v for v in VERBS if " " not in v] + ["гүйлгээ хийх","валют солих","зээл авах","худалдаж авах"]
    random.shuffle(nouns); random.shuffle(verbs)
    for n, v in itertools.islice(itertools.product(nouns, verbs), TARGET_EXPRESSIONS*2):
        exp = f"{n} {v}"
        expressions.append(exp)
    expressions = sorted(set(expressions))[:TARGET_EXPRESSIONS]

with open(os.path.join(BASE_DIR,"expressions.csv"),"w",encoding="utf-8",newline="") as f:
    w=csv.writer(f); w.writerow(["id","expression"])
    for i,exp in enumerate(expressions,1):
        w.writerow([f"{i:04d}",exp])

# ------------ Build numbers_currency.txt ------------
num_lines = []
base_amounts = list(range(1,21)) + [25,30,40,50,60,70,80,90,100,150,200,300,500,700,900,1000,1500,2000,5000,10000,20000,100000,200000,500000,1000000,2000000,5000000,1000000000]
for amt in base_amounts:
    for cur in CURRENCIES:
        num_lines.append(f"{number_words(amt)} {cur}")
# modified thousand forms for 1..9
for n in range(1,10):
    for cur in CURRENCIES:
        num_lines.append(f"{thousand_modified(n)} {cur}")
numbers_currency = sorted(set(num_lines))
with open(os.path.join(BASE_DIR,"numbers_currency.txt"),"w",encoding="utf-8") as f:
    f.write("\n".join(numbers_currency))

# ------------ Sentence templates ------------
SUBJ = ["Би","Ах","Эгч","Найз","Үйлчлүүлэгч","Хэрэглэгч","Менежер","Хүүхэд","Тэр"]
POSSESSIVE = ["миний","ахын","эгчийн","үйлчлүүлэгчийн","хэрэглэгчийн"]
PLACES = ["банк","салбар","кафе","дэлгүүр","гэр","эмнэлэг","сургууль","ажил"]
ACTIONS_BANK = ["дансаа шалгамаар байна","гүйлгээ хийсэн","төлбөр төлөх гэсэн юм","валют сольё гэж байна",
                "зээл авах хүсэлт өгсөн","карт ашигласан","шилжүүлэг амжилттай боллоо","шилжүүлэг амжилтгүй боллоо"]
ACTIONS_TECH = ["QR код уншуулах гэж байна","бар код уншигдсангүй","файл илгээсэн","файл хуулбарласан",
                "нууц үгээ сэргээх хэрэгтэй","апп нээгдсэнгүй","имэйл илгээгдсэн","веб хуудас ачаалсан"]
ACTIONS_DAILY = ["кофе ууж байна","цай ууж сууна","дэлгүүр орлоо","бороо орж байна","салхи хүчтэй байна","кино үзэж байна","ном уншиж байна"]
TIME_PHRASES = TIME + ["одоохон","дараа нь","саяхан","түрүүн","энэ долоо хоногт","дараа сард"]

def random_amount_sentence():
    amt = random.choice(numbers_currency)
    cur = ""  # already includes currency
    subj = random.choice(SUBJ)
    verb = random.choice(["шилжүүллээ","авлаа","хийв","орлоо","хасагдлаа","нэмэгдлээ"])
    obj = random.choice(["дансанд","данснаас","баланс дээр","төлбөрт"])
    return f"{subj} {obj} {amt} {verb}."

def random_bank_sentence():
    subj = random.choice(SUBJ)
    t = random.choice(TIME_PHRASES)
    act = random.choice(ACTIONS_BANK)
    return f"{subj} {t} {act}."

def random_tech_sentence():
    subj = random.choice(SUBJ)
    act = random.choice(ACTIONS_TECH)
    return f"{subj} {act}."

def random_daily_sentence():
    subj = random.choice(SUBJ)
    place = random.choice(PLACES)
    act = random.choice(ACTIONS_DAILY)
    t = random.choice(TIME_PHRASES)
    return f"{subj} {place}-д {t} {act}."

def from_expression_sentence(exp):
    # Expand 2–3 word expression into a clean sentence
    subj = random.choice(SUBJ)
    t = random.choice(["","өнөөдөр","маргааш","одоохон","саяхан"])
    t = (t + " ") if t else ""
    # Ensure proper ending
    if exp.endswith("хийх") or exp.endswith("авах") or exp.endswith("шалгах") or exp.endswith("нээх") or exp.endswith("хаах") or exp.endswith("илгээх"):
        ending = random.choice(["гэсэн юм.","гэж байна.","болно.","хийлээ.","хийв."])
        return f"{subj} {t}{exp} {ending}"
    return f"{subj} {t}{exp}."

# ------------ Generate ~20k sentences (unique, varied) ------------
sent_hash = set()
sentences = []

def push_sentence(s):
    s = " ".join(s.split())  # normalize spaces
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    if h not in sent_hash:
        sent_hash.add(h)
        sentences.append(s)
        return True
    return False

# Seed from expressions (ensures your “one word → many sentences” principle)
for exp in tqdm(expressions, desc="Seeding from expressions"):
    s = from_expression_sentence(exp)
    push_sentence(s)
    if len(sentences) >= TARGET_SENTENCES//4:
        break

# Mix sentence types until we hit TARGET_SENTENCES
generators = [random_bank_sentence, random_tech_sentence, random_daily_sentence, random_amount_sentence]
pbar = tqdm(total=TARGET_SENTENCES, desc="Generating sentences", initial=len(sentences))
while len(sentences) < TARGET_SENTENCES:
    g = random.choice(generators)
    push_sentence(g())
    if random.random() < 0.25 and expressions:
        push_sentence(from_expression_sentence(random.choice(expressions)))
    if random.random() < 0.15:
        # PoV / possessive variation
        poss = random.choice(POSSESSIVE)
        act = random.choice(ACTIONS_BANK + ACTIONS_TECH + ACTIONS_DAILY)
        place = random.choice(PLACES)
        t = random.choice(TIME_PHRASES)
        push_sentence(f"{poss} {place}-д {t} {act}.")
    pbar.n = len(sentences); pbar.refresh()
pbar.close()

# Write sentences.csv
with open(os.path.join(BASE_DIR,"sentences.csv"),"w",encoding="utf-8",newline="") as f:
    w=csv.writer(f); w.writerow(["id","sentence"])
    for i,s in enumerate(sentences,1):
        w.writerow([f"{i:05d}",s])

# Summary print
print("✅ word_list.txt:", len(word_list), "entries")
print("✅ expressions.csv:", len(expressions), "rows")
print("✅ sentences.csv:", len(sentences), "rows (target:", TARGET_SENTENCES, ")")
print("✅ numbers_currency.txt:", len(numbers_currency), "rows")
print("📂 Saved in:", BASE_DIR)
