# Project Presentation Site — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `docs/project_site.html` (single-page RTL Hebrew scrolling presentation) and `server.py` (Flask API) that together showcase the insurance-RAG project during a live presentation.

**Architecture:** A self-contained HTML file (zero external dependencies, works with `file://`) builds all 10 sections with inline CSS/JS. A 35-line Flask server exposes `POST /ask` that calls the existing `src.generation.answer()` and returns JSON — powering the live demo section.

**Tech Stack:** HTML5, CSS3 (custom properties, IntersectionObserver animations, CSS bar chart), vanilla JS (fetch), Flask 3.x, pytest + Flask test client.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `server.py` | Create | Flask API — `POST /ask` → `answer()` → JSON |
| `docs/project_site.html` | Create | Full single-page presentation |
| `tests/test_server.py` | Create | Flask endpoint unit test (mocked `answer()`) |

---

## Task 1: Flask Server (`server.py`)

**Files:**
- Create: `server.py`
- Create: `tests/test_server.py`

- [ ] **Step 1: Install Flask**

```bash
.venv/Scripts/pip install flask flask-cors
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_server.py`:

```python
"""Unit tests for server.py Flask API."""
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def client(monkeypatch):
    """Flask test client with answer() mocked out."""
    import server
    # Patch answer before app is used
    def _fake_answer(question, family_id=None, strategy="section_aware", top_k=5, **kwargs):
        return {
            "answer": "הפרנשייז הוא 3,000 ₪",
            "sources": ["## פרק 3\n\nהפרנשייז..."],
            "strategy": strategy,
            "question": question,
        }
    monkeypatch.setattr(server, "answer", _fake_answer)
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


def test_ask_returns_answer(client):
    resp = client.post(
        "/ask",
        data=json.dumps({"question": "מה הפרנשייז?", "strategy": "section_aware"}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["answer"] == "הפרנשייז הוא 3,000 ₪"
    assert "sources" in data


def test_ask_missing_question_returns_400(client):
    resp = client.post(
        "/ask",
        data=json.dumps({"strategy": "section_aware"}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_ask_defaults_strategy_to_section_aware(client):
    resp = client.post(
        "/ask",
        data=json.dumps({"question": "מה הפרנשייז?"}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    assert resp.get_json()["strategy"] == "section_aware"


def test_cors_header_present(client):
    resp = client.post(
        "/ask",
        data=json.dumps({"question": "שאלה"}),
        content_type="application/json",
    )
    assert "Access-Control-Allow-Origin" in resp.headers
```

- [ ] **Step 3: Run test — expect ImportError (red)**

```bash
.venv/Scripts/python -m pytest tests/test_server.py -v
```

Expected: `ModuleNotFoundError: No module named 'server'`

- [ ] **Step 4: Implement `server.py`**

```python
"""Flask API server for the insurance-RAG live demo.

Run:
    python server.py

Exposes:
    POST /ask  { "question": str, "strategy": str? }
            -> { "answer": str, "sources": [str], "strategy": str }
"""
from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.generation import answer  # noqa: E402

app = Flask(__name__)
CORS(app)  # allow requests from file:// origin


@app.route("/ask", methods=["POST"])
def ask():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "missing question"}), 400

    strategy = body.get("strategy", "section_aware")

    try:
        result = answer(question, strategy=strategy)
        return jsonify({
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "strategy": result.get("strategy", strategy),
        })
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("RAG server running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
```

- [ ] **Step 5: Run tests — expect green**

```bash
.venv/Scripts/python -m pytest tests/test_server.py -v
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: Flask API server for live demo (POST /ask)"
```

---

## Task 2: HTML skeleton + CSS variables + Hero section

**Files:**
- Create: `docs/project_site.html`

- [ ] **Step 1: Create HTML skeleton with CSS variables**

Create `docs/project_site.html` — paste this full skeleton:

```html
<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>מערכת RAG לפוליסות ביטוח בעברית</title>
<style>
/* ── Reset & Variables ─────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:        #0f172a;
  --bg-card:   #1e293b;
  --bg-card2:  #273549;
  --accent:    #818cf8;
  --accent2:   #34d399;
  --accent3:   #f472b6;
  --text:      #f1f5f9;
  --muted:     #94a3b8;
  --border:    #334155;
  --radius:    12px;
  --font:      'Segoe UI', 'Arial', sans-serif;
}
html { scroll-behavior: smooth; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  line-height: 1.7;
  direction: rtl;
}

/* ── Scroll-reveal animation ───────────────────────────── */
.reveal {
  opacity: 0;
  transform: translateY(28px);
  transition: opacity 0.6s ease, transform 0.6s ease;
}
.reveal.visible {
  opacity: 1;
  transform: translateY(0);
}

/* ── Section base ──────────────────────────────────────── */
section {
  max-width: 1100px;
  margin: 0 auto;
  padding: 80px 32px;
}
.section-title {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 12px;
  color: var(--text);
}
.section-title span { color: var(--accent); }
.section-sub {
  color: var(--muted);
  font-size: 1.05rem;
  margin-bottom: 40px;
}
.divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 0;
}

/* ── Sticky nav ────────────────────────────────────────── */
nav {
  position: fixed;
  top: 0; right: 0; left: 0;
  z-index: 100;
  background: rgba(15,23,42,0.92);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: center;
  gap: 8px;
  padding: 10px 20px;
}
nav a {
  color: var(--muted);
  text-decoration: none;
  font-size: 0.85rem;
  padding: 6px 14px;
  border-radius: 20px;
  transition: all 0.2s;
}
nav a:hover { background: var(--bg-card); color: var(--text); }

/* ── Hero ──────────────────────────────────────────────── */
#hero {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
  padding-top: 60px;
  max-width: 100%;
}
.hero-tag {
  display: inline-block;
  background: rgba(129,140,248,0.15);
  color: var(--accent);
  border: 1px solid rgba(129,140,248,0.3);
  border-radius: 20px;
  padding: 6px 18px;
  font-size: 0.85rem;
  margin-bottom: 24px;
  letter-spacing: 0.05em;
}
.hero-title {
  font-size: clamp(2.2rem, 5vw, 3.8rem);
  font-weight: 800;
  line-height: 1.2;
  margin-bottom: 20px;
  background: linear-gradient(90deg, #f1f5f9, #818cf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  color: var(--muted);
  font-size: 1.15rem;
  max-width: 600px;
  margin: 0 auto 40px;
}
.hero-stats {
  display: flex;
  gap: 20px;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 44px;
}
.stat-chip {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 28px;
  text-align: center;
}
.stat-chip .val {
  font-size: 2rem;
  font-weight: 800;
  color: var(--accent);
  display: block;
}
.stat-chip .lbl {
  font-size: 0.8rem;
  color: var(--muted);
}
.hero-btns { display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; }
.btn {
  padding: 12px 28px;
  border-radius: 8px;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
  text-decoration: none;
  transition: all 0.2s;
  border: none;
}
.btn-primary { background: var(--accent); color: #0f172a; }
.btn-primary:hover { background: #a5b4fc; }
.btn-outline { background: transparent; color: var(--text); border: 1px solid var(--border); }
.btn-outline:hover { background: var(--bg-card); }
</style>
</head>
<body>

<nav>
  <a href="#why">הרקע</a>
  <a href="#data">הנתונים</a>
  <a href="#arch">ארכיטקטורה</a>
  <a href="#chunking">חלוקה</a>
  <a href="#goldset">קבוצת הזהב</a>
  <a href="#results">תוצאות</a>
  <a href="#conclusions">מסקנות</a>
  <a href="#run">הרצה</a>
  <a href="#demo">דמו חי</a>
</nav>

<!-- HERO -->
<div id="hero">
  <div class="hero-tag">NLP Project · RAG Pipeline · Hebrew</div>
  <h1 class="hero-title">מערכת RAG לפוליסות<br>ביטוח בעברית</h1>
  <p class="hero-sub">שליפה מוגברת-ייצור על קורפוס פוליסות ביטוח ישראליות —<br>מ-PDF גולמי ועד תשובה מנומקת עם ציטוטים</p>
  <div class="hero-stats">
    <div class="stat-chip"><span class="val">0.720</span><span class="lbl">Hit@5</span></div>
    <div class="stat-chip"><span class="val">0.534</span><span class="lbl">MRR</span></div>
    <div class="stat-chip"><span class="val">50</span><span class="lbl">שאלות זהב</span></div>
    <div class="stat-chip"><span class="val">4</span><span class="lbl">פוליסות</span></div>
  </div>
  <div class="hero-btns">
    <a href="#results" class="btn btn-primary">ראה תוצאות</a>
    <a href="#demo" class="btn btn-outline">דמו חי</a>
    <a href="https://github.com/dudumrk2/insurance-rag" target="_blank" class="btn btn-outline">GitHub ↗</a>
  </div>
</div>
<hr class="divider">

<!-- SECTIONS WILL BE ADDED IN SUBSEQUENT TASKS -->

<script>
// Scroll-reveal
const observer = new IntersectionObserver(
  entries => entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); }),
  { threshold: 0.12 }
);
document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
</script>
</body>
</html>
```

- [ ] **Step 2: Open in browser and verify Hero looks correct**

Open `docs/project_site.html` with `file://` in Chrome. Expected: dark gradient background, large Hebrew title, 4 stat chips, 3 buttons.

- [ ] **Step 3: Commit**

```bash
git add docs/project_site.html
git commit -m "feat: project site — skeleton + Hero section"
```

---

## Task 3: "למה הפרויקט" + "הנתונים" sections

**Files:**
- Modify: `docs/project_site.html` — add two `<section>` blocks and CSS

- [ ] **Step 1: Add CSS for cards and timeline**

Inside the `<style>` block, before `</style>`, append:

```css
/* ── Cards ─────────────────────────────────────────────── */
.cards { display: grid; gap: 20px; }
.cards-2 { grid-template-columns: 1fr 1fr; }
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px 32px;
}
.card-icon { font-size: 2rem; margin-bottom: 12px; }
.card h3 { font-size: 1.2rem; margin-bottom: 8px; color: var(--text); }
.card p { color: var(--muted); font-size: 0.95rem; }

/* ── Table ──────────────────────────────────────────────── */
.data-table { width: 100%; border-collapse: collapse; margin-top: 24px; }
.data-table th { background: var(--bg-card2); padding: 12px 16px; text-align: right; font-size: 0.85rem; color: var(--muted); border-bottom: 1px solid var(--border); }
.data-table td { padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
.data-table tr:last-child td { border-bottom: none; }
code { background: rgba(129,140,248,0.12); color: var(--accent); padding: 2px 7px; border-radius: 4px; font-size: 0.85em; }

/* ── Timeline ───────────────────────────────────────────── */
.timeline {
  display: flex;
  align-items: center;
  gap: 0;
  margin-top: 40px;
  overflow-x: auto;
  padding-bottom: 8px;
}
.tl-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1;
  min-width: 100px;
}
.tl-icon {
  width: 52px; height: 52px;
  border-radius: 50%;
  background: var(--bg-card2);
  border: 2px solid var(--accent);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.4rem;
  position: relative;
  z-index: 1;
}
.tl-label { font-size: 0.78rem; color: var(--muted); margin-top: 8px; text-align: center; }
.tl-arrow { color: var(--border); font-size: 1.4rem; padding: 0 4px; flex-shrink: 0; }
```

- [ ] **Step 2: Add sections HTML — paste before the closing `<script>` tag**

```html
<hr class="divider">

<!-- WHY -->
<section id="why">
  <div class="reveal">
    <h2 class="section-title">למה <span>פוליסות ביטוח</span>?</h2>
    <p class="section-sub">פרויקט עם שתי מטרות — אקדמית ויישומית</p>
    <div class="cards cards-2">
      <div class="card">
        <div class="card-icon">🎓</div>
        <h3>מטרה אקדמית</h3>
        <p>בניית RAG pipeline מלא: קליטה, חלוקה, הטמעות, שליפה, ייצור, קבוצת זהב וניסוי השחלפה. פוליסות ביטוח מכילות ידע חוזי ספציפי ש-LLM לא מכיר — הבחירה האידיאלית להדגמת ערך ה-RAG.</p>
      </div>
      <div class="card">
        <div class="card-icon">🚀</div>
        <h3>מטרה יישומית</h3>
        <p>המנוע עתיד להתחבר ל-<code>ai-wealth-monitor</code> ולאפשר לכל משפחה לשאול שאלות על הפוליסות שלה בזמן אמת, דרך הצ'אט הקיים — ללא שינוי בממשק.</p>
      </div>
    </div>
    <div style="margin-top:24px; background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:20px 28px; border-right: 3px solid var(--accent);">
      <strong>הבעיה שפותרים:</strong> מודלי שפה גדולים (LLMs) <em>מזייפים עובדות</em> כשנשאלים על פרטי פוליסה ספציפיים — תקרות כיסוי, השתתפות עצמית, חריגים — כי הם לא מכירים אותם. RAG פותר זאת על ידי שליפה דינמית של הסעיף הנכון לפני ייצור התשובה.
    </div>
  </div>
</section>
<hr class="divider">

<!-- DATA -->
<section id="data">
  <div class="reveal">
    <h2 class="section-title">הנתונים ו<span>עיבוד מקדים</span></h2>
    <p class="section-sub">4 פוליסות PDF דיגיטליות → Markdown נקי → ChromaDB</p>
    <table class="data-table">
      <thead><tr><th>מסמך</th><th>סוג</th><th>גודל</th><th>הערות</th></tr></thead>
      <tbody>
        <tr><td><code>car_policy</code></td><td>ביטוח רכב</td><td>בינוני</td><td>פוליסה בסיסית</td></tr>
        <tr><td><code>car_policy1</code></td><td>ביטוח רכב</td><td>גדול</td><td>פוליסה מורחבת</td></tr>
        <tr><td><code>health_policy</code></td><td>ביטוח בריאות</td><td>גדול</td><td>השתלות, גמלאות, ניתוחים</td></tr>
        <tr><td><code>home_policy</code></td><td>ביטוח דירה</td><td>בינוני</td><td>תכולה, נזקי טבע, אחריות</td></tr>
      </tbody>
    </table>
    <div class="timeline">
      <div class="tl-step"><div class="tl-icon">📄</div><div class="tl-label">PDF<br>גולמי</div></div>
      <div class="tl-arrow">←</div>
      <div class="tl-step"><div class="tl-icon">⚙️</div><div class="tl-label">Docling<br>PDF→MD</div></div>
      <div class="tl-arrow">←</div>
      <div class="tl-step"><div class="tl-icon">🔒</div><div class="tl-label">הסרת<br>PII</div></div>
      <div class="tl-arrow">←</div>
      <div class="tl-step"><div class="tl-icon">✂️</div><div class="tl-label">חלוקה<br>לקטעים</div></div>
      <div class="tl-arrow">←</div>
      <div class="tl-step"><div class="tl-icon">🧠</div><div class="tl-label">הטמעות<br>e5-large</div></div>
      <div class="tl-arrow">←</div>
      <div class="tl-step"><div class="tl-icon">🗄️</div><div class="tl-label">ChromaDB<br>Index</div></div>
    </div>
  </div>
</section>
<hr class="divider">
```

- [ ] **Step 3: Open in browser — verify 2 sections appear with correct styling**

- [ ] **Step 4: Commit**

```bash
git add docs/project_site.html
git commit -m "feat: project site — Why + Data sections"
```

---

## Task 4: ארכיטקטורה + Chunking sections

**Files:**
- Modify: `docs/project_site.html`

- [ ] **Step 1: Add CSS for pipeline and chunking comparison**

Append to `<style>`:

```css
/* ── Pipeline ───────────────────────────────────────────── */
.pipeline {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-top: 32px;
}
.pipe-step {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 16px;
  text-align: center;
  position: relative;
}
.pipe-step::before {
  content: attr(data-n);
  position: absolute;
  top: -10px; right: 12px;
  background: var(--accent);
  color: #0f172a;
  width: 22px; height: 22px;
  border-radius: 50%;
  font-size: 0.7rem;
  font-weight: 800;
  display: flex; align-items: center; justify-content: center;
}
.pipe-step .icon { font-size: 1.8rem; margin-bottom: 8px; }
.pipe-step h4 { font-size: 0.95rem; margin-bottom: 4px; }
.pipe-step p { font-size: 0.78rem; color: var(--muted); }

/* ── Chunking comparison ────────────────────────────────── */
.chunk-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 28px; }
.chunk-box {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
}
.chunk-box.good { border-color: var(--accent2); }
.chunk-box.bad  { border-color: #f87171; }
.chunk-box h4 { margin-bottom: 4px; font-size: 1rem; }
.chunk-box .tag {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 20px;
  font-size: 0.75rem;
  margin-bottom: 14px;
}
.tag-good { background: rgba(52,211,153,0.15); color: var(--accent2); }
.tag-bad  { background: rgba(248,113,113,0.15); color: #f87171; }
.chunk-text {
  background: var(--bg);
  border-radius: 6px;
  padding: 14px;
  font-size: 0.82rem;
  line-height: 1.6;
  color: var(--muted);
  border: 1px solid var(--border);
  font-family: monospace;
  white-space: pre-wrap;
}
.chunk-text mark { background: rgba(129,140,248,0.25); color: var(--text); border-radius: 2px; }
```

- [ ] **Step 2: Add sections HTML — before the closing `<script>` tag**

```html
<!-- ARCHITECTURE -->
<section id="arch">
  <div class="reveal">
    <h2 class="section-title">ארכיטקטורת <span>המערכת</span></h2>
    <p class="section-sub">Pipeline מלא מ-PDF ועד תשובה — 7 שלבים</p>
    <div class="pipeline">
      <div class="pipe-step" data-n="1">
        <div class="icon">📄</div>
        <h4>PDF גולמי</h4>
        <p>4 פוליסות דיגיטליות</p>
      </div>
      <div class="pipe-step" data-n="2">
        <div class="icon">⚙️</div>
        <h4>Docling</h4>
        <p>PDF → Markdown עם טבלאות</p>
      </div>
      <div class="pipe-step" data-n="3">
        <div class="icon">🔒</div>
        <h4>הסרת PII</h4>
        <p>Regex + מחרוזות ידועות</p>
      </div>
      <div class="pipe-step" data-n="4">
        <div class="icon">✂️</div>
        <h4>Chunking</h4>
        <p>section_aware / fixed_size</p>
      </div>
      <div class="pipe-step" data-n="5">
        <div class="icon">🧠</div>
        <h4>e5-large</h4>
        <p>1024-dim embeddings</p>
      </div>
      <div class="pipe-step" data-n="6">
        <div class="icon">🗄️</div>
        <h4>ChromaDB</h4>
        <p>cosine + family_id filter</p>
      </div>
      <div class="pipe-step" data-n="7">
        <div class="icon">✨</div>
        <h4>Gemini Flash</h4>
        <p>תשובה עם ציטוטים</p>
      </div>
    </div>
  </div>
</section>
<hr class="divider">

<!-- CHUNKING -->
<section id="chunking">
  <div class="reveal">
    <h2 class="section-title">שתי <span>אסטרטגיות חלוקה</span></h2>
    <p class="section-sub">ההשערה: חלוקה לפי סעיפים טבעיים תנצח חלוקה מכנית</p>
    <div class="chunk-compare">
      <div class="chunk-box bad">
        <h4>fixed_size</h4>
        <span class="tag tag-bad">944 קטעים • 500 tokens</span>
        <p style="font-size:0.85rem;color:var(--muted);margin-bottom:12px;">חותך בגבול שרירותי — גורם לפיצול סעיפים באמצע</p>
        <div class="chunk-text">...תנאים נוספים לפוליסה.

## שירות מתן רכ<mark>
────── חתיכה N ──────</mark>
בי חלופי

מתחייב אינו...</div>
      </div>
      <div class="chunk-box good">
        <h4>section_aware</h4>
        <span class="tag tag-good">447 קטעים • ≤700 tokens</span>
        <p style="font-size:0.85rem;color:var(--muted);margin-bottom:12px;">חותך בגבולות כותרת — כל קטע = יחידה סמנטית שלמה</p>
        <div class="chunk-text"><mark>## שירות מתן רכב חלופי</mark>

מתחייב אינו והמבטח
הסדר במוסך חלופי רכב
יינתן לרכב חלקי נזק...
────── קטע שלם ──────</div>
      </div>
    </div>
    <div style="margin-top:16px;text-align:center;color:var(--muted);font-size:0.9rem;">
      💡 כל קטע נושא <code>anchor</code> = 80 התווים הראשונים — מפתח citation יציב בין שתי האסטרטגיות
    </div>
  </div>
</section>
<hr class="divider">
```

- [ ] **Step 3: Open in browser — verify pipeline grid and chunking comparison appear**

- [ ] **Step 4: Commit**

```bash
git add docs/project_site.html
git commit -m "feat: project site — Architecture + Chunking sections"
```

---

## Task 5: קבוצת הזהב + תוצאות ה-Ablation

**Files:**
- Modify: `docs/project_site.html`

- [ ] **Step 1: Add CSS for gold set steps and bar chart**

Append to `<style>`:

```css
/* ── Gold set steps ─────────────────────────────────────── */
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 28px 0; }
.step-box {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px 20px;
  text-align: center;
}
.step-num {
  width: 36px; height: 36px;
  background: var(--accent);
  color: #0f172a;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 1rem;
  margin: 0 auto 12px;
}
.step-box h4 { font-size: 0.95rem; margin-bottom: 6px; }
.step-box p  { font-size: 0.82rem; color: var(--muted); }

/* ── Results table ──────────────────────────────────────── */
.results-table { width: 100%; border-collapse: collapse; margin: 28px 0 0; }
.results-table th { background: var(--bg-card2); padding: 12px 16px; text-align: right; font-size: 0.85rem; color: var(--muted); border-bottom: 1px solid var(--border); }
.results-table td { padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
.results-table tr.best td { background: rgba(129,140,248,0.07); font-weight: 600; }
.results-table tr.best td:first-child { color: var(--accent); }

/* ── Bar chart ──────────────────────────────────────────── */
.bar-chart { margin-top: 36px; }
.bar-chart h4 { color: var(--muted); font-size: 0.85rem; margin-bottom: 16px; font-weight: 500; }
.bar-row { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.bar-label { width: 130px; font-size: 0.85rem; color: var(--muted); text-align: left; flex-shrink: 0; }
.bar-wrap { flex: 1; background: var(--bg-card2); border-radius: 4px; height: 26px; overflow: hidden; }
.bar-fill {
  height: 100%;
  border-radius: 4px;
  background: var(--bg-card2);
  transition: width 1.2s ease;
  display: flex; align-items: center; padding-right: 8px;
  font-size: 0.78rem; font-weight: 600;
}
.bar-fill.accent  { background: linear-gradient(90deg, var(--accent), #a5b4fc); color: #0f172a; }
.bar-fill.muted   { background: var(--border); color: var(--muted); }
.bar-val { width: 50px; font-size: 0.85rem; text-align: left; flex-shrink: 0; }
```

- [ ] **Step 2: Add sections HTML — before the closing `<script>` tag**

```html
<!-- GOLD SET -->
<section id="goldset">
  <div class="reveal">
    <h2 class="section-title">קבוצת <span>הזהב</span></h2>
    <p class="section-sub">50 שאלות-תשובות שנבחרו ידנית מ-75 מועמדים שנוצרו ע"י Gemini</p>
    <div class="steps">
      <div class="step-box">
        <div class="step-num">1</div>
        <h4>ייצור אוטומטי</h4>
        <p>Gemini 2.5 Flash יצר 75 זוגות שאלה-תשובה-ציטוט — 15–25 לכל מסמך</p>
      </div>
      <div class="step-box">
        <div class="step-num">2</div>
        <h4>ביקורת ידנית</h4>
        <p>כלי HTML אינטראקטיבי לסינון ובחירת 50 השאלות הטובות ביותר</p>
      </div>
      <div class="step-box">
        <div class="step-num">3</div>
        <h4>עיגון ל-Anchor</h4>
        <p>כל שאלה מקושרת לקטע המקורי בפוליסה דרך 80 תווי ה-anchor</p>
      </div>
    </div>
    <div style="text-align:center;margin-top:20px;">
      <a href="../eval/selector.html" target="_blank" class="btn btn-outline" style="display:inline-flex;align-items:center;gap:8px;">
        🔍 צפה ב-75 השאלות המקוריות ↗
      </a>
    </div>
    <div style="margin-top:28px;display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
      <div class="stat-chip"><span class="val" style="font-size:1.4rem;">11</span><span class="lbl">car_policy</span></div>
      <div class="stat-chip"><span class="val" style="font-size:1.4rem;">14</span><span class="lbl">car_policy1</span></div>
      <div class="stat-chip"><span class="val" style="font-size:1.4rem;">12</span><span class="lbl">health_policy</span></div>
      <div class="stat-chip"><span class="val" style="font-size:1.4rem;">13</span><span class="lbl">home_policy</span></div>
    </div>
  </div>
</section>
<hr class="divider">

<!-- RESULTS -->
<section id="results">
  <div class="reveal">
    <h2 class="section-title">תוצאות <span>ניסוי ההשחלפה</span></h2>
    <p class="section-sub">4 תצורות × 50 שאלות × top_k=5</p>
    <table class="results-table">
      <thead><tr><th>תצורה</th><th>קטעים</th><th>Hit@1</th><th>Hit@3</th><th>Hit@5</th><th>MRR</th></tr></thead>
      <tbody>
        <tr class="best"><td>section_aware ⭐</td><td>447</td><td>0.400</td><td>0.660</td><td>0.720</td><td>0.534</td></tr>
        <tr><td>fixed_300</td><td>1,700</td><td>0.180</td><td>0.240</td><td>0.300</td><td>0.216</td></tr>
        <tr><td>fixed_700</td><td>656</td><td>0.080</td><td>0.220</td><td>0.280</td><td>0.150</td></tr>
        <tr><td>fixed_500</td><td>944</td><td>0.040</td><td>0.200</td><td>0.280</td><td>0.119</td></tr>
      </tbody>
    </table>
    <div class="bar-chart">
      <h4>Hit@5 — שיעור שאלות עם קטע נכון בתוצאות top-5</h4>
      <div class="bar-row">
        <div class="bar-label">section_aware</div>
        <div class="bar-wrap"><div class="bar-fill accent" style="width:72%">72%</div></div>
        <div class="bar-val">0.720</div>
      </div>
      <div class="bar-row">
        <div class="bar-label">fixed_300</div>
        <div class="bar-wrap"><div class="bar-fill muted" style="width:30%">30%</div></div>
        <div class="bar-val">0.300</div>
      </div>
      <div class="bar-row">
        <div class="bar-label">fixed_700</div>
        <div class="bar-wrap"><div class="bar-fill muted" style="width:28%">28%</div></div>
        <div class="bar-val">0.280</div>
      </div>
      <div class="bar-row">
        <div class="bar-label">fixed_500</div>
        <div class="bar-wrap"><div class="bar-fill muted" style="width:28%">28%</div></div>
        <div class="bar-val">0.280</div>
      </div>
    </div>
    <div class="bar-chart" style="margin-top:24px;">
      <h4>MRR — ממוצע הדדי של הדירוג (גבוה יותר = קרוב יותר למקום 1)</h4>
      <div class="bar-row">
        <div class="bar-label">section_aware</div>
        <div class="bar-wrap"><div class="bar-fill accent" style="width:53.4%">0.534</div></div>
        <div class="bar-val">0.534</div>
      </div>
      <div class="bar-row">
        <div class="bar-label">fixed_300</div>
        <div class="bar-wrap"><div class="bar-fill muted" style="width:21.6%">0.216</div></div>
        <div class="bar-val">0.216</div>
      </div>
      <div class="bar-row">
        <div class="bar-label">fixed_700</div>
        <div class="bar-wrap"><div class="bar-fill muted" style="width:15%">0.150</div></div>
        <div class="bar-val">0.150</div>
      </div>
      <div class="bar-row">
        <div class="bar-label">fixed_500</div>
        <div class="bar-wrap"><div class="bar-fill muted" style="width:11.9%">0.119</div></div>
        <div class="bar-val">0.119</div>
      </div>
    </div>
  </div>
</section>
<hr class="divider">
```

- [ ] **Step 3: Open in browser — verify gold set steps, table, and bar charts render**

- [ ] **Step 4: Commit**

```bash
git add docs/project_site.html
git commit -m "feat: project site — Gold Set + Ablation Results sections"
```

---

## Task 6: מסקנות + הרצה sections

**Files:**
- Modify: `docs/project_site.html`

- [ ] **Step 1: Add CSS for conclusions and code tabs**

Append to `<style>`:

```css
/* ── Conclusions ────────────────────────────────────────── */
.conclusions { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 28px; }
.concl-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px 24px;
  text-align: center;
}
.concl-icon { font-size: 2.4rem; margin-bottom: 14px; }
.concl-card h3 { font-size: 1rem; margin-bottom: 8px; color: var(--accent); }
.concl-card p { font-size: 0.88rem; color: var(--muted); line-height: 1.6; }

/* ── Code tabs ──────────────────────────────────────────── */
.code-tabs { margin-top: 28px; }
.tab-buttons { display: flex; gap: 4px; margin-bottom: -1px; }
.tab-btn {
  padding: 8px 18px;
  border: 1px solid var(--border);
  border-bottom: none;
  border-radius: 8px 8px 0 0;
  background: var(--bg-card2);
  color: var(--muted);
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s;
}
.tab-btn.active { background: var(--bg-card); color: var(--text); border-color: var(--border); }
.tab-panel {
  display: none;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 0 8px 8px 8px;
  position: relative;
}
.tab-panel.active { display: block; }
.tab-panel pre {
  padding: 20px 24px;
  overflow-x: auto;
  font-size: 0.83rem;
  line-height: 1.6;
  color: #e2e8f0;
  direction: ltr;
  text-align: left;
}
.copy-btn {
  position: absolute;
  top: 10px; left: 12px;
  background: var(--bg-card2);
  border: 1px solid var(--border);
  color: var(--muted);
  border-radius: 6px;
  padding: 4px 12px;
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s;
}
.copy-btn:hover { color: var(--text); }
```

- [ ] **Step 2: Add sections HTML — before the closing `<script>` tag**

```html
<!-- CONCLUSIONS -->
<section id="conclusions">
  <div class="reveal">
    <h2 class="section-title">מה <span>למדנו</span></h2>
    <p class="section-sub">שלוש תובנות מרכזיות מהפרויקט</p>
    <div class="conclusions">
      <div class="concl-card">
        <div class="concl-icon">🏆</div>
        <h3>section_aware מנצחת</h3>
        <p>על מסמכים עם מבנה כותרות ברור — חלוקה לפי סעיפים טבעיים עולה פי 2.5 ב-MRR על כל וריאנט fixed_size</p>
      </div>
      <div class="concl-card">
        <div class="concl-icon">🔑</div>
        <h3>תחיליות e5 — קריטי</h3>
        <p><code>"passage: "</code> על קטעים ו-<code>"query: "</code> על שאילתות — השמטתן גורמת לירידה מורגשת בלי שגיאה גלויה. טעות שקטה.</p>
      </div>
      <div class="concl-card">
        <div class="concl-icon">⚓</div>
        <h3>anchor > chunk_id</h3>
        <p>80 תווים ראשונים כמפתח ציטוט שורדים בין שתי האסטרטגיות — מאפשרים השוואה הוגנת עם אותה קבוצת זהב</p>
      </div>
    </div>
  </div>
</section>
<hr class="divider">

<!-- RUN -->
<section id="run">
  <div class="reveal">
    <h2 class="section-title">הרצת <span>המערכת</span></h2>
    <p class="section-sub">מ-clone עד תשובה חיה — 5 פקודות</p>
    <div class="code-tabs">
      <div class="tab-buttons">
        <button class="tab-btn active" onclick="showTab('install')">התקנה</button>
        <button class="tab-btn" onclick="showTab('index')">בניית Index</button>
        <button class="tab-btn" onclick="showTab('eval')">הרצת Eval</button>
        <button class="tab-btn" onclick="showTab('query')">שאילתה</button>
        <button class="tab-btn" onclick="showTab('server')">שרת דמו</button>
      </div>
      <div id="tab-install" class="tab-panel active">
        <button class="copy-btn" onclick="copyCode('install')">העתק</button>
        <pre id="code-install">git clone https://github.com/dudumrk2/insurance-rag.git
cd insurance-rag
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[all]"
echo GEMINI_API_KEY=your_key > .env</pre>
      </div>
      <div id="tab-index" class="tab-panel">
        <button class="copy-btn" onclick="copyCode('index')">העתק</button>
        <pre id="code-index">python scripts/pdf_to_md.py    # PDF → Markdown
python scripts/redact.py       # הסרת PII
python build_index.py          # embedding + ChromaDB
# → indices/insurance_section_aware  (447 chunks)
# → indices/insurance_fixed          (944 chunks)</pre>
      </div>
      <div id="tab-eval" class="tab-panel">
        <button class="copy-btn" onclick="copyCode('eval')">העתק</button>
        <pre id="code-eval">python eval/run_eval.py --out eval/ablation_results.md
# section_aware  Hit@5=0.720  MRR=0.534
# fixed_500      Hit@5=0.280  MRR=0.119
# fixed_300      Hit@5=0.300  MRR=0.216
# fixed_700      Hit@5=0.280  MRR=0.150</pre>
      </div>
      <div id="tab-query" class="tab-panel">
        <button class="copy-btn" onclick="copyCode('query')">העתק</button>
        <pre id="code-query">python -c "
from src.generation import answer
result = answer('מה הפרנשייז על נזק מלא לרכב?',
                strategy='section_aware')
print(result['answer'])
print('מקורות:', result['sources'])
"</pre>
      </div>
      <div id="tab-server" class="tab-panel">
        <button class="copy-btn" onclick="copyCode('server')">העתק</button>
        <pre id="code-server">python server.py
# RAG server running on http://localhost:5000
# POST /ask  {"question": "...", "strategy": "section_aware"}
# ←          {"answer": "...", "sources": [...]}</pre>
      </div>
    </div>
  </div>
</section>
<hr class="divider">
```

- [ ] **Step 3: Add tab-switching JS — inside the `<script>` block, after the IntersectionObserver code**

```js
function showTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}
function copyCode(name) {
  const text = document.getElementById('code-' + name).textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = event.target;
    btn.textContent = '✓ הועתק';
    setTimeout(() => btn.textContent = 'העתק', 1800);
  });
}
```

- [ ] **Step 4: Open in browser — verify tabs switch correctly and copy works**

- [ ] **Step 5: Commit**

```bash
git add docs/project_site.html
git commit -m "feat: project site — Conclusions + Run Commands sections"
```

---

## Task 7: דמו חי (Live Demo section)

**Files:**
- Modify: `docs/project_site.html`

- [ ] **Step 1: Add CSS for demo section**

Append to `<style>`:

```css
/* ── Live Demo ──────────────────────────────────────────── */
#demo { background: linear-gradient(180deg, var(--bg) 0%, #1e1b4b22 100%); }
.demo-box {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 36px 40px;
  max-width: 720px;
  margin: 0 auto;
}
.demo-box label { display: block; font-size: 0.85rem; color: var(--muted); margin-bottom: 8px; }
.demo-input {
  width: 100%;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 16px;
  color: var(--text);
  font-size: 1rem;
  font-family: var(--font);
  direction: rtl;
  margin-bottom: 16px;
  transition: border-color 0.2s;
}
.demo-input:focus { outline: none; border-color: var(--accent); }
.demo-controls { display: flex; gap: 12px; align-items: center; margin-bottom: 20px; flex-wrap: wrap; }
.strategy-select {
  background: var(--bg);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 0.9rem;
  cursor: pointer;
}
.ask-btn {
  background: var(--accent);
  color: #0f172a;
  border: none;
  border-radius: 8px;
  padding: 11px 28px;
  font-size: 0.95rem;
  font-weight: 700;
  cursor: pointer;
  transition: background 0.2s;
}
.ask-btn:hover { background: #a5b4fc; }
.ask-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.demo-spinner {
  display: none;
  width: 20px; height: 20px;
  border: 2px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.demo-result {
  display: none;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 20px 24px;
  margin-top: 4px;
}
.demo-answer { font-size: 1rem; line-height: 1.7; margin-bottom: 14px; }
.demo-sources { border-top: 1px solid var(--border); padding-top: 12px; }
.demo-sources h5 { font-size: 0.8rem; color: var(--muted); margin-bottom: 8px; }
.source-chip {
  display: inline-block;
  background: rgba(129,140,248,0.1);
  border: 1px solid rgba(129,140,248,0.2);
  color: var(--accent);
  border-radius: 6px;
  padding: 3px 10px;
  font-size: 0.75rem;
  margin: 2px;
  direction: rtl;
}
.demo-error {
  display: none;
  background: rgba(248,113,113,0.1);
  border: 1px solid rgba(248,113,113,0.3);
  border-radius: 8px;
  padding: 14px 18px;
  color: #f87171;
  font-size: 0.9rem;
  margin-top: 4px;
}
.suggested-qs { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.sug-q {
  background: var(--bg-card2);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 6px 14px;
  font-size: 0.82rem;
  color: var(--muted);
  cursor: pointer;
  transition: all 0.2s;
}
.sug-q:hover { border-color: var(--accent); color: var(--text); }
```

- [ ] **Step 2: Add demo HTML — before the closing `</body>` tag, after the last `<hr>`**

```html
<!-- LIVE DEMO -->
<section id="demo">
  <div class="reveal">
    <h2 class="section-title" style="text-align:center;">דמו <span>חי</span></h2>
    <p class="section-sub" style="text-align:center;">שאל שאלה על הפוליסות — מקבל תשובה אמיתית מה-RAG</p>
    <div class="demo-box">
      <label>שאלה בעברית</label>
      <textarea class="demo-input" id="q-input" rows="2"
        placeholder="לדוגמה: מה הפרנשייז על נזק מלא לרכב?"></textarea>
      <div class="demo-controls">
        <select class="strategy-select" id="strategy-select">
          <option value="section_aware">section_aware (מומלץ)</option>
          <option value="fixed">fixed_500</option>
        </select>
        <button class="ask-btn" id="ask-btn" onclick="askQuestion()">שאל את הפוליסה ✦</button>
        <div class="demo-spinner" id="spinner"></div>
      </div>
      <div class="demo-result" id="demo-result">
        <div class="demo-answer" id="demo-answer"></div>
        <div class="demo-sources" id="demo-sources">
          <h5>📎 מקורות</h5>
          <div id="sources-list"></div>
        </div>
      </div>
      <div class="demo-error" id="demo-error"></div>
      <div>
        <div style="font-size:0.8rem;color:var(--muted);margin-bottom:8px;">שאלות לדוגמה:</div>
        <div class="suggested-qs">
          <span class="sug-q" onclick="fillQ(this)">מה הפרנשייז על נזק מלא לרכב?</span>
          <span class="sug-q" onclick="fillQ(this)">האם יש כיסוי לגניבת רכב?</span>
          <span class="sug-q" onclick="fillQ(this)">מה גמלת ההחלמה לאחר השתלה?</span>
          <span class="sug-q" onclick="fillQ(this)">מה מכסה ביטוח הדירה לנזקי מים?</span>
          <span class="sug-q" onclick="fillQ(this)">מהו סכום הפיצוי לאחר ניתוח מורכב?</span>
        </div>
      </div>
    </div>
  </div>
</section>

<footer style="text-align:center;padding:40px 20px;color:var(--muted);font-size:0.85rem;border-top:1px solid var(--border);">
  insurance-rag · built with multilingual-e5-large + ChromaDB + Gemini 2.5 Flash ·
  <a href="https://github.com/dudumrk2/insurance-rag" target="_blank" style="color:var(--accent);text-decoration:none;">GitHub ↗</a>
</footer>
```

- [ ] **Step 3: Add demo JS — inside the `<script>` block, after the tab functions**

```js
const API = 'http://localhost:5000/ask';

async function askQuestion() {
  const q = document.getElementById('q-input').value.trim();
  if (!q) return;
  const strategy = document.getElementById('strategy-select').value;

  document.getElementById('ask-btn').disabled = true;
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('demo-result').style.display = 'none';
  document.getElementById('demo-error').style.display = 'none';

  try {
    const res = await fetch(API, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, strategy })
    });
    if (!res.ok) throw new Error(`שגיאת שרת ${res.status}`);
    const data = await res.json();

    document.getElementById('demo-answer').textContent = data.answer;
    const srcList = document.getElementById('sources-list');
    srcList.innerHTML = '';
    (data.sources || []).forEach(s => {
      const chip = document.createElement('span');
      chip.className = 'source-chip';
      chip.textContent = s.slice(0, 60) + (s.length > 60 ? '…' : '');
      srcList.appendChild(chip);
    });
    document.getElementById('demo-result').style.display = 'block';
  } catch (e) {
    const errBox = document.getElementById('demo-error');
    errBox.style.display = 'block';
    errBox.innerHTML = e.message.includes('Failed to fetch')
      ? '⚠️ השרת אינו זמין — הרץ <code>python server.py</code> ואז נסה שוב'
      : '⚠️ ' + e.message;
  } finally {
    document.getElementById('ask-btn').disabled = false;
    document.getElementById('spinner').style.display = 'none';
  }
}

function fillQ(el) {
  document.getElementById('q-input').value = el.textContent;
}

document.getElementById('q-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); askQuestion(); }
});
```

- [ ] **Step 4: Open in browser and test demo (needs `python server.py` running)**

```bash
# Terminal 1:
python server.py
# Terminal 2: open docs/project_site.html in browser
# Type a question, click "שאל את הפוליסה", verify answer appears
```

- [ ] **Step 5: Commit**

```bash
git add docs/project_site.html
git commit -m "feat: project site — Live Demo section with Flask API integration"
```

---

## Task 8: Final polish + PR

**Files:**
- Modify: `docs/project_site.html` — minor polish
- Modify: `pyproject.toml` — add flask dependency

- [ ] **Step 1: Add Flask to pyproject.toml**

In `pyproject.toml`, add a `server` extra:

```toml
server = ["flask>=3.0", "flask-cors>=4.0"]
```

And add `"flask>=3.0", "flask-cors>=4.0"` to the `all` list as well.

- [ ] **Step 2: Run all tests**

```bash
.venv/Scripts/python -m pytest tests/ -v
```

Expected: all tests pass (including `tests/test_server.py`)

- [ ] **Step 3: Final commit + push**

```bash
git add pyproject.toml docs/project_site.html
git commit -m "feat: add flask server extra to pyproject.toml"
git push
```

- [ ] **Step 4: Open PR**

```bash
gh pr create \
  --title "feat: project presentation site + Flask demo server" \
  --body "Adds docs/project_site.html (10-section RTL single-page presentation) and server.py (Flask POST /ask). Opens with file://, live demo requires python server.py."
```

---

## Self-Review

**Spec coverage:**
- ✅ Hero section with stats
- ✅ Why section (2 cards)
- ✅ Data section (corpus table + timeline)
- ✅ Architecture (pipeline steps)
- ✅ Chunking (side-by-side comparison)
- ✅ Gold Set (3-step process + link to selector.html)
- ✅ Ablation Results (table + CSS bar chart)
- ✅ Conclusions (3 cards)
- ✅ Run Commands (tabbed code blocks + copy)
- ✅ Live Demo (fetch → Flask → answer())
- ✅ server.py with POST /ask
- ✅ CORS header
- ✅ Tests for server.py

**No placeholders:** All code steps contain actual code. No "TBD" or "similar to above."

**Type consistency:** `answer()` imported consistently in `server.py`. JS uses consistent IDs (`demo-result`, `demo-error`, `sources-list`). `showTab(name)` matches `onclick="showTab('install')"`.
