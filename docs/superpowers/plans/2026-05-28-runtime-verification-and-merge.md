# Runtime Verification And PR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the updated `insurance-rag` repository end to end, then create a PR for `docs/align-current-system-status`. Do not merge; this user can push branches and open PRs only.

**Architecture:** Treat `docs/align-current-system-status` as a docs/metadata accuracy branch and verify it against the current runtime. Validate in layers: repository state, dependency metadata, tests, Flask server, `/ask` API behavior, and optional embedding/eval paths.

**Tech Stack:** Python 3.10+, pytest, Flask, ChromaDB, sentence-transformers/e5, Google Gemini via `google-genai`, git/GitHub.

---

### Task 1: Restore Session State

**Files:**
- Read: `README.md`
- Read: `docs/report.md`
- Read: `docs/superpowers/plans/2026-05-28-runtime-verification-and-merge.md`

- [ ] **Step 1: Fetch remote branches**

Run:

```bash
git fetch --all --prune
```

Expected: remote branch `origin/docs/align-current-system-status` exists.

- [ ] **Step 2: Check out the docs branch**

Run:

```bash
git switch docs/align-current-system-status
git status --short --branch
```

Expected:

```text
## docs/align-current-system-status...origin/docs/align-current-system-status
```

- [ ] **Step 3: Confirm the branch head**

Run:

```bash
git log --oneline -3
```

Expected: top commit is:

```text
3d5a94c docs: align docs with current system status
```

### Task 2: Verify Docs And Metadata Accuracy

**Files:**
- Read: `README.md`
- Read: `pyproject.toml`
- Read: `docs/design-spec.md`
- Read: `docs/DESIGN_RATIONALE.md`
- Read: `docs/report.md`
- Read: `docs/project_site.html`
- Read: `docs/roadmap.html`

- [ ] **Step 1: Scan for stale high-risk claims**

Run:

```bash
rg -n 'anthropic|Claude.*gold|gold.*Claude|500 tokens|700 tokens|≤700 tokens|rag_system|VectorStore|FixedSizeChunker|SectionAwareChunker|test_embeddings|test_vector_store|test_e2e|chunk_ids cited|\[chunk_id|parse citations|family_id.*mandatory|raises without|early ValueError|Q1.*פנסים|Q4.*גניבה חלקית|not implemented|not started' README.md pyproject.toml docs/design-spec.md docs/DESIGN_RATIONALE.md docs/report.md docs/project_site.html docs/roadmap.html
```

Expected: no matches.

- [ ] **Step 2: Validate Python project metadata**

Run:

```bash
python -B -c "import tomllib; data=tomllib.load(open('pyproject.toml','rb')); assert data['project']['optional-dependencies']['goldset'] == ['google-genai>=0.8']; assert 'flask>=3.0' in data['project']['optional-dependencies']['dev']; assert 'anthropic' not in str(data).lower(); print('pyproject metadata ok')"
```

Expected:

```text
pyproject metadata ok
```

- [ ] **Step 3: Check formatting whitespace**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

### Task 3: Install Test Dependencies And Run Tests

**Files:**
- Read: `pyproject.toml`
- Read: `tests/test_server.py`

- [ ] **Step 1: Install fast-test dependencies**

Run:

```bash
pip install -e ".[dev]"
```

Expected: pytest, Flask, and flask-cors are installed in the active environment.

- [ ] **Step 2: Run the fast suite including server tests**

Run:

```bash
python -m pytest -q -m "not slow"
```

Expected:

```text
89 passed, 4 deselected
```

If the exact count changes because new tests were added, verify there are 0 failures and 0 errors before proceeding.

- [ ] **Step 3: Verify the public `answer()` contract with mocked retrieval/generation**

Run:

```bash
python -B -c "from src.generation import answer; res=answer('q', _retrieve_fn=lambda **kw:[{'text':'passage: body','anchor':'a','source_doc':'doc','chunk_id':'id','score':0.9,'section':''}], _generate_fn=lambda prompt:'ok'); print(sorted(res.keys())); print(res['sources'])"
```

Expected:

```text
['answer', 'question', 'sources', 'strategy']
['a']
```

### Task 4: Verify Flask Runtime

**Files:**
- Read: `server.py`
- Read: `docs/project_site.html`

- [ ] **Step 1: Start the server**

Run in a long-running terminal:

```bash
python server.py
```

Expected: server starts on `http://localhost:5000`.

- [ ] **Step 2: Smoke-test validation behavior**

Run in a second terminal:

```bash
curl -s -X POST http://localhost:5000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"","strategy":"section_aware"}'
```

Expected: HTTP 400 response body indicating the question is required.

- [ ] **Step 3: Smoke-test mocked or indexed `/ask` behavior**

Run:

```bash
curl -s -X POST http://localhost:5000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"מה הפרנשייז על נזק מלא לרכב?","strategy":"section_aware"}'
```

Expected: JSON object with `answer`, `sources`, and `strategy`.

If this fails with missing Chroma collections, run Task 5 before retrying.

- [ ] **Step 4: Stop the server cleanly**

Run:

```bash
pkill -f "python server.py"
```

Expected: no lingering `python server.py` process.

### Task 5: Verify Index And Retrieval Evaluation

**Files:**
- Read: `build_index.py`
- Read: `eval/run_eval.py`
- Read: `eval/ablation_results.md`

- [ ] **Step 1: Check whether persistent indices exist**

Run:

```bash
ls -la indices
```

Expected: directory exists with Chroma collections. If it does not exist, continue to Step 2.

- [ ] **Step 2: Build indices when absent**

Run:

```bash
pip install -e ".[embeddings,vectorstore,generation]"
python build_index.py
```

Expected: `indices/` is created and Chroma collections `insurance_fixed` and `insurance_section_aware` are available.

- [ ] **Step 3: Run retrieval eval**

Run:

```bash
python eval/run_eval.py --out eval/ablation_results.md
```

Expected: `eval/ablation_results.md` is regenerated with a `section_aware` row close to the documented metrics. Because Chroma HNSW is approximate, compare directionality rather than requiring byte-identical output: `section_aware` should remain far above fixed variants.

### Task 6: Optional Live Gemini Verification

**Files:**
- Read: `src/generation.py`
- Read: `src/config.py`

- [ ] **Step 1: Confirm API key availability**

Run:

```bash
python -B -c "import os; from dotenv import load_dotenv; load_dotenv(); print(bool(os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')))"
```

Expected:

```text
True
```

If output is `False`, stop live Gemini verification and ask the user to provide a key. Do not claim live generation is verified.

- [ ] **Step 2: Run one live answer**

Run:

```bash
python -B -c "from src.generation import answer; res=answer('מה הפרנשייז על נזק מלא לרכב?', strategy='section_aware'); print(res['answer']); print(res['sources'][:2])"
```

Expected: Hebrew answer text and at least one retrieved source anchor.

### Task 7: Create PR Only

**Files:**
- Read: `README.md`
- Read: `docs/report.md`
- Read: `docs/project_site.html`
- Read: `docs/roadmap.html`

- [ ] **Step 1: Confirm final branch status**

Run:

```bash
git status --short --branch
```

Expected: clean working tree on `docs/align-current-system-status`.

- [ ] **Step 2: Push any new verification-plan commit**

Run:

```bash
git push
```

Expected: remote branch is up to date.

- [ ] **Step 3: Create a PR**

Run:

```bash
gh pr create \
  --base master \
  --head docs/align-current-system-status \
  --title "docs: align docs with current system status" \
  --body "Aligns README, report, project site, roadmap, historical design docs, and dependency metadata with the current implemented RAG system. Verified stale-claim scan, pyproject metadata, whitespace, and fast non-server tests before handoff."
```

Expected: GitHub returns a PR URL.

If `gh pr create` fails due authentication or permissions, use the pushed branch URL shown by `git push`: `https://github.com/dudumrk2/insurance-rag/pull/new/docs/align-current-system-status`.

---

## Handoff Notes

- Current branch: `docs/align-current-system-status`
- Pushed branch: `origin/docs/align-current-system-status`
- Current docs commit before this plan file: `3d5a94c docs: align docs with current system status`
- Known limitation from this session: full `pytest -m "not slow"` originally failed because Flask was absent; the branch now adds Flask and flask-cors to the `dev` extra, so the next session should install `.[dev]` before re-running the full fast suite.
- Do not claim the application is fully working until Tasks 3-6 are executed in the new session.
