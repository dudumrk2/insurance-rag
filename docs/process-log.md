# יומן תהליך — ביצוע הפרויקט

> מסמך זה מסכם את כל השלבים שבוצעו עד כה בפרויקט RAG על פוליסות ביטוח עבריות,
> כולל החלטות טכניות, תוצאות, ובעיות שנפתרו בדרך.

---

## סקירת הפרויקט

**מטרה כפולה:**
1. עבודת אמצע אקדמית — מערכת RAG עם ablation study על שתי אסטרטגיות chunking.
2. אינטגרציה עתידית לאפליקציה `ai-wealth-monitor` דרך משתמש הדמו.

**Stack טכנולוגי מאושר:**
- Embeddings: `intfloat/multilingual-e5-large` (sentence-transformers)
- Generation: `gemini-2.5-flash`
- Vector store: ChromaDB
- PDF → Markdown: Docling

**ריפו:** `github.com/dudumrk2/insurance-rag` (sibling ל-`ai-wealth-monitor`, לא מקונן בתוכו)

---

## הקורפוס

ארבעה קובצי PDF של פוליסות ביטוח אמיתיות (עברית דיגיטלית, לא סרוקות):

| קובץ | תיאור |
|------|--------|
| `car_policy.pdf` | פוליסת הרכב המלאה |
| `car_policy1.pdf` | מסמך כיסויים ועלויות לרכב (54 עמודים) |
| `health_policy.pdf` | פוליסת בריאות קבוצתית (אמדוקס) |
| `home_policy.pdf` | פוליסת דירה |

הקבצים ב-`data/raw/` — **gitignored לעולם**, מכילים PII אמיתי.

---

## Step 1 — המרת PDF למארקדאון וניקוי PII

### 1א. המרה עם Docling

**כלי:** Docling (ML-based layout model, lazy import ב-`src/pdf_to_md.py`)

**בעיות שנתקלנו ופתרונות:**

| בעיה | גורם | פתרון |
|------|-------|--------|
| `WinError 1314` בקובץ הראשון | Docling ניסה ליצור symlinks בהורדת המודלים — ב-Windows דורש הרשאות Admin | הפעלה חד-פעמית, המודל נשמר ב-cache |
| `std::bad_alloc` + segfault (24+ עמודים אבודים) | Pipeline ברירת מחדל: `queue_max_size=100`, 4 threads, batch של 4 — שחיקת RAM | כוונון memory-light (ראה טבלה) |
| OCR של סינית על טקסט עברי | Docling טוען OCR מודלים של CJK כברירת מחדל | `do_ocr=False` (PDFים דיגיטליים, לא נדרש OCR) |

**הגדרות pipeline סופיות (`src/pdf_to_md.py`):**

```python
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True
pipeline_options.generate_page_images = False
pipeline_options.layout_batch_size = 1      # (ברירת מחדל: 4)
pipeline_options.table_batch_size = 1
pipeline_options.queue_max_size = 2          # (ברירת מחדל: 100)
pipeline_options.accelerator_options.num_threads = 1
pipeline_options.table_structure_options.mode = TableFormerMode.FAST
```

**סיבה:** הפחתת שימוש ב-RAM על ידי עיבוד עמוד אחד בכל פעם עם תור זעיר.  
**תוצאה:** כל 4 הקבצים הומרו בהצלחה ללא אבדן עמודים.

### 1ב. ניקוי PII (`src/redaction.py` + `scripts/redact.py`)

**שכבה 1 — ביטויים רגולריים (regex):**

| סוג | ביטוי | Placeholder |
|-----|--------|-------------|
| תעודת זהות ישראלית | `\b\d{9}\b` | `[ת"ז]` |
| טלפון | `\b0(?:5\d\|[2-4]\|7\d\|8\|9)-?\d{7}\b` | `[טלפון]` |
| אימייל | `\b[\w.+-]+@[\w-]+\.[\w.-]+\b` | `[אימייל]` |
| לוחית רישוי | `\b\d{2,3}-\d{3}-\d{2,3}\b` | `[רישוי]` |
| IBAN ישראלי | `\bIL\d{2}(?:\s?\d){19}\b` | `[IBAN]` |

> **תיקון:** הביטוי המקורי ל-IBAN היה שגוי (22 ספרות במקום 21). תוקן ל-`(?:\s?\d){19}`.

**שכבה 2 — מחרוזות ידועות (`data/known_pii.json`):**
רשימת שמות, מספרי טלפון, אימיילים וכתובת — נמחקים בהתאמה מדויקת (ארוכים ראשונה).  
הקובץ **gitignored** — PII אמיתי בטוח בו.

**תהליך הניקוי המלא:**
1. Docling ממיר PDF → Markdown
2. Pass 1: regex מוחק מזהים סטנדרטיים
3. Pass 2: מחרוזות ידועות (שמות, כתובות) נמחקות לפי סדר אורך יורד
4. log מציין רק הקשר **אחרי** הניקוי — אין דליפת PII ל-log
5. קבצי Markdown נכתבים ל-`data/redacted/`

**תוצאות — ריצה סופית (56 הסרות):**

| קובץ | גודל | שורות | הסרות | סוגים שנמצאו |
|------|------|-------|--------|---------------|
| `car_policy.md` | 20 KB | 313 | 11 | אימייל ×2, לוחית רישוי ×1, שמות ×7, כתובת ×1 |
| `car_policy1.md` | 204 KB | 1,267 | 3 | לוחית רישוי ×1, טלפון ×2 |
| `health_policy.md` | 407 KB | 2,261 | 30 | אימייל ×11, טלפון ×10, ת"ז ×2, שמות ×7 |
| `home_policy.md` | 27 KB | 450 | 12 | אימייל ×2, טלפון ×5, שמות ×4, כתובת ×1 |
| **סה"כ** | **658 KB** | **4,291** | **56** | |

> הכתובת `כפר סבא` התגלתה בביקורת ידנית לאחר הריצה הראשונה — נוספה ל-`known_pii.json` והריצה בוצעה מחדש.

**11 בדיקות TDD ל-`src/redaction.py`** — כולן ירוקות.

---

## Step 2 — Chunking (שני אסטרטגיות)

### מבנה Chunk

כל chunk הוא מילון Python / שורת JSON עם השדות:

| שדה | תיאור |
|-----|--------|
| `chunk_id` | מזהה ייחודי: `{family_id}_{strategy}_{doc_name}_{idx}` |
| `text` | `"passage: " + raw_text` (e5 prefix נדרש לאיכות retrieval) |
| `source_doc` | שם הקובץ ללא סיומת |
| `strategy` | `"fixed"` או `"section_aware"` |
| `family_id` | מפתח multi-tenancy (ברירת מחדל: `demo_family_001`) |
| `anchor` | 80 התווים הראשונים של הטקסט הגולמי — עוגן יציב ל-gold set |
| `section` | כותרת `##` הקרובה ביותר (רק ב-section_aware; אחרת `null`) |

> **e5 prefix:** מודל `multilingual-e5-large` דורש `"passage: "` על chunks ו-`"query: "` על שאילתות — קריטי לאיכות ה-embedding.

### אסטרטגיה 1 — Fixed-size

- חלוקה לחלונות של **500 תווים** עם **50 תווים overlap**
- פשוט, אחיד, עובד על כל מבנה מסמך
- Output: `data/processed/chunks_fixed.jsonl`

### אסטרטגיה 2 — Section-aware

- פיצול על כותרות `##` שמייצר Docling
- מגבלה: **700 tokens** (~2800 תווים) לסעיף; סעיפים ארוכים מתחלקים ב-fixed-size fallback
- שומר הקשר סמנטי — chunk מכיל מידע שייך לאותו נושא
- Output: `data/processed/chunks_section_aware.jsonl`

### תוצאות Chunking

| אסטרטגיה | סה"כ chunks | הערות |
|-----------|-------------|--------|
| `fixed` | **944** | health_policy לבדה: 607 chunks |
| `section_aware` | **447** | כמעט חצי — sections מאחדות תוכן |

**18 בדיקות TDD ל-`src/chunking.py`** — כולן ירוקות (31 בדיקות סה"כ בפרויקט).

---

## מבנה הפרויקט (נוכחי)

```
insurance-rag/
├── data/
│   ├── raw/                  ← PDFs מקוריים (gitignored, PII)
│   ├── redacted/             ← Markdown מנוקה (4 קבצים, מחויב)
│   │   ├── car_policy.md
│   │   ├── car_policy1.md
│   │   ├── health_policy.md
│   │   └── home_policy.md
│   ├── processed/            ← JSONL chunks (2 קבצים, מחויב)
│   │   ├── chunks_fixed.jsonl
│   │   └── chunks_section_aware.jsonl
│   ├── known_pii.json        ← PII אמיתי (gitignored)
│   ├── redaction_log.json    ← סיכום הסרות (ללא PII)
│   └── MANIFEST.md           ← פרטי הקורפוס
├── src/
│   ├── config.py             ← קבועים ונתיבים
│   ├── redaction.py          ← ניקוי PII (regex + known strings)
│   ├── pdf_to_md.py          ← עטיפת Docling
│   └── chunking.py           ← שתי אסטרטגיות chunking
├── scripts/
│   ├── redact.py             ← CLI: PDF → redacted MD
│   └── chunk.py              ← CLI: MD → chunks JSONL
├── tests/
│   ├── test_redaction.py     ← 11 בדיקות
│   ├── test_redact_cli.py    ← בדיקות CLI
│   └── test_chunking.py      ← 18 בדיקות
└── docs/
    ├── design-spec.md
    ├── implementation-plan.md
    └── process-log.md        ← מסמך זה
```

---

## שלבים הבאים

| שלב | תיאור | קבצים |
|-----|--------|--------|
| **Step 3** | Embedding + ChromaDB index | `src/embedder.py`, `build_index.py` |
| **Step 4** | Retrieval (`retrieve()`) | `src/retrieval.py` |
| **Step 5** | Generation (`answer()`) | `src/generation.py` |
| **Step 6** | Gold set (50 שאלות) | `eval/gold_set.jsonl` |
| **Step 7** | Ablation study | `eval/run_eval.py` |
| **Step 8** | דוח 4 עמודים | `docs/report.md` |
