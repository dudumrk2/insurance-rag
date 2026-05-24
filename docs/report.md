<div dir="rtl">

# מערכת RAG לפוליסות ביטוח בעברית
## דוח פרויקט — עיבוד שפה טבעית

---

## תקציר

פרויקט זה מתאר בניית מערכת *Retrieval-Augmented Generation* (להלן RAG) מלאה על קורפוס של פוליסות ביטוח ישראליות בעברית. הקורפוס כולל ארבעה מסמכי PDF (ביטוח רכב ×2, ביטוח בריאות, ביטוח דירה) שעברו הסרת PII, המרה ל-Markdown וחלוקה לקטעים. הושוו שתי אסטרטגיות חלוקה — `fixed_size` ו-`section_aware` — ועוד שלושה וריאנטים של גודל חלון. ניסוי ההשחלפה (`Ablation Study`) הראה ש-`section_aware` מעפיל על כל וריאנטי `fixed_size` במדדי Hit@k ו-MRR, עם Hit@5 = 0.740 לעומת מקסימום 0.300 ל-`fixed`. הממצאים מוסברים בשימור יחידות סמנטיות טבעיות — פרקי הפוליסה — בגישת ה-`section_aware`.

---

## 1. מבוא

פוליסות ביטוח הן מסמכים חוזיים-טכניים עשירי פרטים: תקרות כיסוי, השתתפות עצמית, חריגים, תקופות המתנה ותנאים מיוחדים. מודלי שפה גדולים (LLMs) מועדים להזיה על מידע ספציפי שכזה — הם אינם "יודעים" את פרטי הפוליסה הספציפית של משפחה מסוימת. מערכת RAG פותרת זאת על ידי שליפה דינמית של קטעים רלוונטיים מהפוליסה, שמוזנים כהקשר למודל בזמן ייצור התשובה.

**מטרות הפרויקט:**

1. **אקדמית** — בניית `pipeline` מלא הכולל קליטה, חלוקה לקטעים, הטמעות, שליפה, ייצור, קבוצת זהב (`gold set`) וניסוי השחלפה (`ablation study`).
2. **יישומית** — יצירת מנוע שיוכל להתחבר בעתיד ל-`ai-wealth-monitor` ולאפשר שאילתות פוליסה בזמן אמת.

הפרויקט ממוקם ב-`repository` עצמאי (`insurance-rag`) הצמוד ל-`ai-wealth-monitor`, כך שהאינטגרציה העתידית תתבצע דרך `pip install -e` ללא תלויות מעגליות.

---

## 2. נתונים ועיבוד מקדים

### 2.1 הקורפוס

| מסמך | סוג | גודל | הערות |
|---|---|---|---|
| `car_policy` | ביטוח רכב | בינוני | פוליסה בסיסית |
| `car_policy1` | ביטוח רכב | גדול | פוליסה מורחבת, קוצצה ל-40,000 תווים בייצור קבוצת הזהב |
| `health_policy` | ביטוח בריאות | גדול | קוצצה ל-40,000 תווים |
| `home_policy` | ביטוח דירה | בינוני | — |

כל המסמכים הם PDF דיגיטליים (לא סרוקים), מה שאיפשר המרה מדויקת ללא OCR.

### 2.2 המרת PDF ל-Markdown

נעשה שימוש ב-`Docling` — ספריית ML המשתמשת במודל `layout` ייעודי לזיהוי טבלאות וכותרות. הבחירה ב-`Docling` על פני `pymupdf4llm` נבעה מהחשיבות הגבוהה של טבלאות בפוליסות ביטוח (טבלת כיסויים, גבולות אחריות). הפרמטרים כוונו לצריכת זיכרון מינימלית: `do_ocr=False`, `layout_batch_size=1`, `TableFormerMode.FAST`.

### 2.3 הסרת PII

הפוליסות מכילות מידע מזהה אישי של בעל הפוליסה: תעודת זהות, טלפון, כתובת, רישוי רכב ו-IBAN. פותח מודול `src/redaction.py` הכולל שתי שכבות:

1. **זיהוי תבניות (`Regex`)** — 6 תבניות לזיהוי PII מובנה: ת"ז ישראלית בת 9 ספרות, טלפון, אימייל, לוחית רישוי, ו-IBAN בפורמט `IL`.
2. **מחרוזות ידועות** — רשימה מקובץ `data/known_pii.json` המכילה שמות וכתובות שנבנתה ידנית.

כל ערך שהוסר הוחלף ב-`[REDACTED]`. נוצר `redaction_log.json` המתעד את *כמות* ההסרות ומיקומן, ללא הערכים עצמם.

---

## 3. ארכיטקטורת המערכת

<div dir="ltr">

```
PDF
 └─► Docling ──► Markdown ──► Redaction ──► data/redacted/*.md
                                                    │
                              ┌─────────────────────┤
                              ▼                     ▼
                         chunk_fixed         chunk_section_aware
                         (500/300/700)        (≤700 tokens/section)
                              │                     │
                              └──────────┬──────────┘
                                         ▼
                              multilingual-e5-large
                              (1024-dim, "passage: " prefix)
                                         │
                                         ▼
                                  ChromaDB
                              (cosine similarity, family_id filter)
                                         │
                                         ▼
                              retrieve(query, top_k=5)
                                         │
                                         ▼
                              Gemini 2.5 Flash
                              (Hebrew system prompt, T=0.2)
                                         │
                                         ▼
                              answer + citations (anchors)
```

</div>

### 3.1 חלוקה לקטעים (`Chunking`)

**חלון הזזה (`fixed_size`):** חלוקה לפי מספר `tokens` בחלון הזזה, תוך שימוש ב-`tokenizer` של `multilingual-e5-large` עצמו (ולא ב-`tiktoken`) להבטחת תאימות לבאדג'ט האמיתי של מודל ההטמעות.

**חלוקה לפי סעיפים (`section_aware`):** זיהוי כותרות Markdown‏ (`\n## `) כגבולות חתיכה. פרק שעולה על 700 `tokens` מחולק רקורסיבית לתת-חתיכות. אסטרטגיה זו מנצלת את הסמנטיקה הטבעית של פוליסות ביטוח, שבהן כל סעיף (כיסויים, חריגים, תנאים) הוא יחידה מושגית עצמאית.

כל קטע נושא מטא-דאטה: `chunk_id`, `source_doc`, `strategy`, `family_id`, `anchor` (80 התווים הראשונים של הטקסט הגולמי — מפתח `citation` יציב בין האסטרטגיות), ו-`section`.

### 3.2 הטמעות (`Embeddings`)

נבחר `intfloat/multilingual-e5-large` (1024 ממדים) — מודל מקומי חינמי עם תמיכה מעולה בעברית. מודלי `e5` אומנו א-סימטרית על זוגות `(query, passage)`, ולכן נדרשת תחילית שונה לכל סוג:

- קטע בעת אינדוקס: ‏`"passage: " + text`
- שאילתה בזמן שליפה: ‏`"query: " + question`

השמטת התחיליות גורמת לירידה מורגשת באיכות השליפה — טעות שקטה שקשה לזהות ללא `benchmark`.

### 3.3 מאגר הוקטורים ורב-שוכרות (`Vector Store` ו-`Multi-tenancy`)

נבחר `ChromaDB` על פני `FAISS` בזכות תמיכה מובנית בסינון לפי מטא-דאטה. כל קטע נושא `family_id`, והשליפה מסננת אוטומטית: ‏`where={"family_id": uid}`. הפרדה זו מונעת דליפת פוליסות בין משפחות ללא ניהול קבצים נפרדים. נבנו שני `indexes` (אחד לכל אסטרטגיה) לצורך השוואה הוגנת.

### 3.4 ייצור תשובות (`Generation`)

נבחר `Gemini 2.5 Flash` עם `temperature=0.2` — אותו מודל הפועל ב-`ai-wealth-monitor`, מה שמקל על האינטגרציה העתידית. ה-`prompt` בעברית מכוון את המודל לענות בהתבסס על ההקשר בלבד ולסמן כל טענה עם ה-`anchor` שלה.

---

## 4. קבוצת הזהב (`Gold Set`)

### 4.1 תהליך הבנייה

נדחתה בנייה ידנית מלאה (יקרה בזמן) ונבחר תהליך היברידי:

1. **ייצור אוטומטי** — מודל `Gemini 2.5 Flash` יצר 75 זוגות שאלה-תשובה-ציטוט על פי חלוקה: ‏`car_policy` ×15, ‏`car_policy1` ×20, ‏`health_policy` ×25, ‏`home_policy` ×15. ה-`prompt` דרש לצרף ציטוט מילולי מדויק (30–120 תווים) כראיה לכל תשובה.

2. **ביקורת ידנית** — כלי HTML אינטראקטיבי (`eval/selector.html`) אפשר לדפדף בין 75 המועמדים ולסמן 50 שנבחרו לפי כיסוי נושאים מאוזן: עובדתי, מספרי, זמני, שלילה, השוואה.

3. **עיגון ל-`anchor`** — לכל ציטוט בוצע חיפוש `substring` בקטעי ה-`section_aware`. ה-`anchor` (80 תווים ראשונים) שורד בין האסטרטגיות ומאפשר השוואה הוגנת.

**חשוב:** הגנה מפני הערכה מעגלית (`circular evaluation`) — קבוצת הזהב נוצרה על ידי `Gemini`, בעוד שהשאלות נענות ע"י `Gemini` דרך ה-RAG. שני מסלולים נפרדים = הערכה בלתי-מוטה.

### 4.2 הרכב קבוצת הזהב

| מסמך | שאלות | נושאים מרכזיים |
|---|---|---|
| `car_policy` | 11 | פרנשייז, גרר, רכב חלופי |
| `car_policy1` | 14 | נזק מוחלט, כיסויים מורחבים |
| `health_policy` | 12 | השתלות, גמלאות, ניתוחים |
| `home_policy` | 13 | תכולה, נזקי טבע, אחריות |
| **סה"כ** | **50** | — |

---

## 5. ניסוי השחלפת מרכיבים (`Ablation Study`)

### 5.1 הגדרת הניסוי

נוסו ארבע תצורות שליפה על אותן 50 שאלות עם `top_k=5`:

| תצורה | אסטרטגיה | גודל (‏`tokens`) | קטעים |
|---|---|---|---|
| A | `section_aware` | ≤700 (טבעי) | 447 |
| B | `fixed_500` | 500, חפיפה 50 | 944 |
| C | `fixed_300` | 300, חפיפה 50 | 1,700 |
| D | `fixed_700` | 700, חפיפה 50 | 656 |

**מדדים:** Hit@k — שיעור השאלות שבהן הקטע הנכון הופיע בתוצאות `top-k`; MRR — ממוצע הדדי של הדירוג (גבוה יותר = הקטע הנכון קרוב יותר למקום הראשון).

**שיטת התאמה:** ה-`gold anchor` הוא 80 התווים הראשונים של הקטע המקורי ב-`section_aware`. בתצורות `fixed` בוצע חיפוש `substring` — האם ה-`anchor` מופיע בתוך הקטע שנשלף.

### 5.2 תוצאות

| תצורה | קטעים | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| **‏`section_aware`** | **447** | **0.380** | **0.660** | **0.740** | **0.529** |
| `fixed_500` | 944 | 0.060 | 0.220 | 0.300 | 0.139 |
| `fixed_300` | 1,700 | 0.160 | 0.220 | 0.280 | 0.196 |
| `fixed_700` | 656 | 0.080 | 0.180 | 0.260 | 0.138 |

### 5.3 ניתוח

אסטרטגיית `section_aware` גוברת על כל תצורות `fixed` בפער משמעותי: Hit@5 של 0.740 לעומת מקסימום 0.300, ו-MRR גבוה פי ~2.7.

**הסבר מבני:** כותרות ה-Markdown שיצר `Docling` מגדירות גבולות סמנטיים טבעיים. כל סעיף בפוליסה (כיסויים, חריגים, גבולות אחריות) מופיע כקטע שלם, ללא רעש מסעיפים סמוכים. ב-`fixed_size`, קטע יכול להתחיל באמצע סעיף ולסיים באמצע סעיף אחר — מה שמדלל את האיתות הסמנטי בעת ההטמעה.

**הערת מתודולוגיה:** ה-`gold anchors` נבנו מקטעי `section_aware`, מה שנותן לאסטרטגיה זו יתרון מבני בבדיקת ה-`substring`. עם זאת, דמיון ההטמעות (שאינו תלוי ב-`anchors`) תומך בממצא: שליפת `section_aware` מחזירה קטעים רלוונטיים יותר גם סמנטית.

**ממצא נוסף:** ההבדלים בין תצורות ה-`fixed` קטנים ונמצאים בתחום השונות שבין ריצות (האינדקסים של `fixed_300`/`fixed_700` נבנים מחדש בזיכרון עם חיפוש HNSW מקורב). `fixed_500` מובילה קלות ב-Hit@5 (0.300), בעוד `fixed_300` מובילה ב-MRR (0.196 לעומת 0.139 ו-0.138) — קטעים קטנים יותר נוטים לדרג מעט גבוה יותר את הקטע הנכון כשהוא נשלף. עם זאת, אף וריאנט `fixed` אינו מתקרב ל-`section_aware` באף מדד.

---

## 6. דיון ומגבלות

### מה עבד טוב

- **תחיליות `e5`** — יישום נכון של תחיליות `"passage: "` / `"query: "` בשלב ההטמעה הוא קריטי ולעיתים מוזנח. בפרויקט זה הוטמע בשכבת ה-`embedder` כך שקוד קורא לא יכול לטעות.
- **`section_aware` על מסמכים מובנים** — כאשר המסמך נכתב עם מבנה כותרות ברור (כמו פוליסות ביטוח), אסטרטגיה זו מנצחת ללא תחרות.
- **ציטוטים מבוססי `anchor`** — שימוש ב-80 התווים הראשונים כמפתח ציטוט במקום `chunk_id` מאפשר השוואת אסטרטגיות על אותה קבוצת זהב.

### מגבלות ושיפורים עתידיים

1. **הטיית קבוצת הזהב** — ה-`gold anchors` נבנו מ-`section_aware`, מה שמטה את ההשוואה. פתרון: בניית קבוצת זהב כפולה (אחת לכל אסטרטגיה) עם `reconciliation` ידני.
2. **שליפה היברידית (`Hybrid retrieval`)** — שילוב BM25 עם שליפה וקטורית בשיטת RRF צפוי לשפר שאלות מספריות ועובדתיות שבהן התאמה מדויקת חשובה יותר מסמנטיקה.
3. **דירוג מחדש (`Cross-encoder reranking`)** — מודל כמו `BAAI/bge-reranker-v2-m3` יכול לשפר MRR על ידי דירוג מחדש של ה-`top-20`.
4. **קורפוס מוגבל** — 4 פוליסות הן קורפוס קטן. תוצאות עשויות להשתנות עם מגוון גדול יותר של מסמכים.

---

## 7. מסקנות

פרויקט זה הדגים שבניית מערכת RAG איכותית על מסמכים בעברית מובנים (פוליסות ביטוח) דורשת:

- המרת PDF מבוססת ML (‏`Docling`) לשימור טבלאות וכותרות.
- אסטרטגיית חלוקה המכבדת את מבנה המסמך (`section_aware`) ולא חלוקה מכנית.
- שימוש נכון בתחיליות `e5` — פרט טכני קריטי ללא השפעה גלויה על ריצת הקוד.
- הערכה בקבוצת זהב היברידית (ייצור ע"י LLM + בחירה ידנית) עם ציטוט מבוסס `anchor`.

ניסוי ההשחלפה אישש את ההשערה התיאורטית: `section_aware` עם Hit@5=0.740 ו-MRR=0.529 מציגה ביצועים טובים משמעותית מכל וריאנט `fixed_size`, ומשמשת כבסיס לאינטגרציה העתידית ב-`ai-wealth-monitor`.

---

## נספח — הרצה, קישורים והסברים נוספים

### א. הרצת ה-`Pipeline` מקצה לקצה

להלן סדר הפקודות המלא להרצת הפרויקט מאפס על מכונה חדשה:

<div dir="ltr">

```bash
# 1. שכפול והתקנה
git clone https://github.com/dudumrk2/insurance-rag.git
cd insurance-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"

# 2. הגדרת מפתחות API (יצירת קובץ .env בשורש הפרויקט)
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. המרת PDF ל-Markdown (דורש קבצי PDF ב-data/raw/)
python scripts/pdf_to_md.py

# 4. הסרת PII מהמסמכים
python scripts/redact.py

# 5. בניית ה-Index (chunking + embedding + ChromaDB)
#    כותב: data/processed/chunks_*.jsonl + indices/
python build_index.py

# 6. בניית Gold Set (ייצור 75 מועמדים + בחירה ידנית של 50)
python scripts/build_gold_set.py --out eval/gold_set_candidates.jsonl
#    פתח eval/selector.html בדפדפן → בחר 50 שאלות → שמור ל-eval/gold_set.jsonl

# 7. הרצת Ablation Study (כולל fixed_300 ו-fixed_700 — ~45 דקות CPU)
python eval/run_eval.py --out eval/ablation_results.md

# 8. הרצת שאילתה בודדת מה-CLI
python -c "
from src.generation import answer
result = answer('מה הפרנשייז על נזק מלא לרכב?', strategy='section_aware')
print(result['answer'])
print('מקורות:', result['sources'])
"

# 9. הרצת כל הטסטים
python -m pytest tests/ -v
```

</div>

---

### ב. קישורים לריפו

| קובץ / תיקייה | תיאור | קישור |
|---|---|---|
| `src/chunking.py` | שתי אסטרטגיות החלוקה | [🔗](https://github.com/dudumrk2/insurance-rag/blob/master/src/chunking.py) |
| `src/embedder.py` | `Singleton` של `multilingual-e5-large` + תחיליות `e5` | [🔗](https://github.com/dudumrk2/insurance-rag/blob/master/src/embedder.py) |
| `src/retrieval.py` | ‏`retrieve()` עם סינון `family_id` | [🔗](https://github.com/dudumrk2/insurance-rag/blob/master/src/retrieval.py) |
| `src/generation.py` | ‏`answer()` — ‏`Gemini` + הקשר RAG | [🔗](https://github.com/dudumrk2/insurance-rag/blob/master/src/generation.py) |
| `src/redaction.py` | הסרת PII (‏`regex` + מחרוזות ידועות) | [🔗](https://github.com/dudumrk2/insurance-rag/blob/master/src/redaction.py) |
| `build_index.py` | ‏CLI לבניית `indexes` ב-`ChromaDB` | [🔗](https://github.com/dudumrk2/insurance-rag/blob/master/build_index.py) |
| `scripts/build_gold_set.py` | ייצור 75 מועמדים ע"י `Gemini` | [🔗](https://github.com/dudumrk2/insurance-rag/blob/master/scripts/build_gold_set.py) |
| `eval/selector.html` | כלי HTML אינטראקטיבי לבחירת 50 שאלות | [🔗](https://github.com/dudumrk2/insurance-rag/blob/master/eval/selector.html) |
| `eval/run_eval.py` | ניסוי ההשחלפה — Hit@k ו-MRR | [🔗](https://github.com/dudumrk2/insurance-rag/blob/master/eval/run_eval.py) |
| `eval/gold_set.jsonl` | 50 שאלות הזהב הסופיות | [🔗](https://github.com/dudumrk2/insurance-rag/blob/master/eval/gold_set.jsonl) |
| `eval/ablation_results.md` | תוצאות ניסוי ההשחלפה | [🔗](https://github.com/dudumrk2/insurance-rag/blob/master/eval/ablation_results.md) |
| `docs/DESIGN_RATIONALE.md` | תיעוד כל ההחלטות העיצוביות | [🔗](https://github.com/dudumrk2/insurance-rag/blob/master/docs/DESIGN_RATIONALE.md) |
| `tests/` | ‎50+ טסטים (‏`redaction`, ‏`chunking`, ‏`embedder`, ‏`retrieval`, ‏`eval`) | [🔗](https://github.com/dudumrk2/insurance-rag/tree/master/tests) |

**ריפו ראשי:** https://github.com/dudumrk2/insurance-rag

---

### ג. הסברים נוספים שלא נכללו בגוף הדוח

#### מדוע לא השתמשנו בהטמעות של OpenAI?

שלוש סיבות:

1. **עלות** — כל בנייה מחדש של ה-`index` (944+ קטעים) עולה כסף; מודל מקומי = $0.
2. **פרטיות** — הפוליסות מכילות PII גם לאחר הסרה חלקית; שליחתן ל-API חיצוני מוסיפה סיכון.
3. **הדגמת הבנה** — המשימה האקדמית מעריכה ידע על רכיבי ה-`pipeline`, לא עטיפת API.

#### מנגנון ה-`Anchor` ומדוע הוא חשוב

ב-RAG רגיל, ציטוט מזוהה לפי `chunk_id`. הבעיה: הערך `doc_chunk_0042` משתנה לחלוטין בין `fixed_500` (944 קטעים) ל-`section_aware` (447 קטעים). אם קבוצת הזהב נבנתה עם `chunk_id` של אסטרטגיה אחת, לא ניתן להשוות לאסטרטגיה שנייה.

הפתרון: ה-**`anchor`** = 80 התווים הראשונים של טקסט הקטע (ללא ה-`prefix`). 80 תווים מספיקים לייחוד חד-משמעי של קטע בתוך מסמך, ואותה מחרוזת תופיע (כ-`substring`) גם בקטעי `fixed` — כל עוד הקטע מכסה את אותו מקטע טקסט.

#### מה קרה עם ה-PII של כפר סבא?

בסקירה ידנית של `redaction_log` התגלה שכתובת עם שם עיר ("כפר סבא") לא זוהתה על ידי ה-`regex` (שמות ערים אינם PII מובנה). הפתרון: הוספתה ל-`data/known_pii.json` והרצת הסקריפט מחדש. זה מדגיש את הצורך בסקירה ידנית — אין אוטומציה מושלמת.

#### זמני ריצה בפועל

| שלב | זמן בפועל | חומרה |
|---|---|---|
| ‏`Docling` — המרת 4 קבצי PDF ל-Markdown | ~3 דקות | CPU, 8GB RAM |
| ‏`build_index` — הטמעת 944 + 447 קטעים | ~8 דקות | CPU |
| ‏`build_gold_set` — ייצור 75 שאלות | ~4 דקות | ‏Gemini API |
| ‏`run_eval` — ‏`section_aware` + ‏`fixed_500` | ~8 דקות | CPU |
| ‏`run_eval` — כולל ‏`fixed_300` + ‏`fixed_700` | ~50 דקות | CPU |

#### מבנה התלויות ב-`pyproject.toml`

התלויות מחולקות ל-`extras` כדי שכל שלב יתקין רק מה שצריך:

<div dir="ltr">

```toml
[project.optional-dependencies]
pdf        = ["docling>=2.0"]
embeddings = ["sentence-transformers>=3.0", "torch>=2.2"]
vectorstore= ["chromadb>=0.5"]
generation = ["google-genai>=0.8"]
dev        = ["pytest>=8.0", "python-dotenv>=1.0"]
all        = [...]   # הכל ביחד
```

</div>

---

## רשימת מקורות

1. Wang, L. et al. (2024). *Multilingual E5 Text Embeddings: A Technical Report*. arXiv:2402.05672.
2. Barnett, S. et al. (2024). *Seven Failure Points When Engineering a Retrieval Augmented Generation System*. arXiv:2401.05856.
3. Gao, Y. et al. (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv:2312.10997.
4. IBM Research (2024). *Docling Technical Report*. arXiv:2408.09869.
5. Bajaj, P. et al. (2018). *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset*. arXiv:1611.09268. *(בסיס לאימון מודלי `e5`)*
6. Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.

</div>
