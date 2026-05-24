# Design Spec — Project Presentation Site

**Date:** 2026-05-24  
**Status:** Approved

---

## Goal

A single-page scrolling HTML presentation (`docs/project_site.html`) + a lightweight Flask server (`server.py`) that together showcase the full insurance-RAG project during a live presentation.

---

## Deliverables

| File | Purpose |
|---|---|
| `docs/project_site.html` | Self-contained HTML (no CDN), opens with `file://` |
| `server.py` | Flask server, `localhost:5000`, powers the live demo section |

---

## Audience & Style

- **Audience:** Live presentation audience
- **Format:** Single-page vertical scroll
- **Style:** Presentation-first — bold gradient hero, large text, subtle CSS scroll animations (`@keyframes` + `IntersectionObserver`), dark background with vibrant accent colours

---

## Page Sections (top → bottom)

### 1. Hero
- Full-viewport height
- Dark-to-blue gradient background
- Large RTL Hebrew title: "מערכת RAG לפוליסות ביטוח בעברית"
- Subtitle: pipeline tagline
- 3 stat chips: Hit@5 = 0.720 / MRR = 0.534 / 50 שאלות זהב
- 3 nav anchor buttons: תוצאות / דמו / קוד

### 2. למה הפרויקט
- 2 side-by-side cards: אקדמי vs יישומי
- Short paragraph: why insurance policies (LLM hallucination problem)

### 3. הנתונים ועיבוד מקדים
- Corpus table (4 docs, type, size)
- Visual horizontal timeline: PDF → PII Redaction → Chunking → Embedding → ChromaDB

### 4. ארכיטקטורה
- Graphical pipeline: 6 numbered steps with arrows, icons per step
- Steps: PDF / Docling / Redact / Chunk / e5-large / ChromaDB / Gemini

### 5. אסטרטגיות חלוקה (Chunking)
- Side-by-side visual comparison
- Left: `fixed_size` — shows text cut arbitrarily mid-sentence
- Right: `section_aware` — shows clean section boundary
- Real Hebrew text snippet from a policy used as example

### 6. קבוצת הזהב
- 3-step visual process: Gemini generates 75 → selector.html → 50 curated
- Button linking to `../eval/selector.html` (opens in new tab)
- Small table: doc breakdown (car×11, health×12, etc.)

### 7. תוצאות ה-Ablation
- Results table (4 rows × 6 cols)
- Bar chart (pure CSS, no canvas): Hit@5 and MRR bars per configuration
- `section_aware` bar highlighted in accent colour

### 8. מסקנות
- 3 large cards with icon + bold heading + 2-line explanation:
  1. section_aware מנצחת על מסמכים מובנים
  2. תחיליות e5 — קריטי ושקוף
  3. anchor > chunk_id לצורך השוואה הוגנת

### 9. הרצה
- Tabbed code blocks: התקנה / build_index / run_eval / שאילתה בודדת
- Copy-to-clipboard button per block

### 10. דמו חי
- Text area: "הקלד שאלה בעברית..."
- Strategy selector: `section_aware` / `fixed_500`
- "שאל את הפוליסה" button
- Loading spinner (CSS)
- Result card: answer text + sources list (anchor strings)
- Error state: "שרת לא זמין — הרץ `python server.py` תחילה"

---

## `server.py` Spec

```
POST /localhost:5000/ask
Body:  { "question": str, "strategy": str }
Response: { "answer": str, "sources": [str], "strategy": str }
CORS: Access-Control-Allow-Origin: *
Error: 500 + { "error": str }
```

- ~30 lines
- Uses `src.generation.answer()`
- Loads ChromaDB indices once on startup (not per-request)
- Requires `.env` with `GEMINI_API_KEY`

---

## Technical Constraints

- HTML: zero external dependencies, works offline (except demo section)
- All CSS inline in `<style>` block
- All JS inline in `<script>` block
- RTL: `<html dir="rtl" lang="he">`
- Scroll animations: `IntersectionObserver` + CSS `opacity/transform` transitions
- Bar chart: pure CSS `width` percentages, no canvas/SVG
- Relative path to `selector.html`: `../eval/selector.html`

---

## Out of Scope

- Mobile responsiveness (presentation is on a laptop/projector)
- Multi-language toggle
- Saving demo answers
