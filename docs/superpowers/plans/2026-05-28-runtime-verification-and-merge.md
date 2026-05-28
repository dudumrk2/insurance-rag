# Docs-Only Verification And PR Review Plan

**Goal:** Verify PR #22 / `docs/align-current-system-status` as a documentation-only
accuracy update against `origin/master`. Do not merge automatically.

**Scope:** This PR should update README and docs only. Runtime code, dependency metadata,
and application behavior are intentionally out of scope. Known dependency metadata cleanup
(`pyproject.toml` / `requirements.txt`) should be handled in a later PR.

---

## Task 1: Restore Repository State

Run:

```bash
git fetch --all --prune
git switch docs/align-current-system-status
git status --short --branch
git log --oneline origin/master..HEAD
```

Expected:

- Branch is `docs/align-current-system-status`.
- Working tree is clean before review.
- PR commits are ahead of `origin/master`.

---

## Task 2: Confirm Docs-Only Diff

Run:

```bash
git diff --name-only origin/master...HEAD
git diff --stat origin/master...HEAD
```

Expected changed files:

```text
README.md
docs/DESIGN_RATIONALE.md
docs/design-spec.md
docs/project_site.html
docs/report.md
docs/roadmap.html
docs/superpowers/plans/2026-05-28-runtime-verification-and-merge.md
```

`pyproject.toml`, `requirements.txt`, `src/`, `server.py`, tests, and data files should
not be part of the PR diff.

---

## Task 3: Scan For Stale Documentation Claims

Run:

```bash
rg -n 'scripts/pdf_to_md\.py|chunks_fixed_size|eval/build_gold_set|eval/results|chroma_fixed_size|chroma_section_aware|build_index\.py --reset|500 tokens|700 tokens|≤700 tokens|rag_system|VectorStore|FixedSizeChunker|SectionAwareChunker|test_embeddings|test_vector_store|test_e2e|chunk_ids cited|\[chunk_id|parse citations|family_id.*mandatory|raises without|early ValueError|Q1.*פנסים|Q4.*גניבה חלקית|not implemented|not started' README.md docs/design-spec.md docs/DESIGN_RATIONALE.md docs/report.md docs/project_site.html docs/roadmap.html
```

Expected: no matches.

Do not scan `pyproject.toml` or `requirements.txt` for this docs-only PR; dependency
metadata is intentionally unchanged and has known stale entries.

---

## Task 4: Verify Formatting

Run:

```bash
git diff --check origin/master...HEAD
```

Expected: no output and exit code 0.

---

## Task 5: Optional Runtime Sanity Checks

Runtime checks are useful, but they are not a merge gate for this docs-only PR because
the branch intentionally does not change dependency metadata or application code.

If the current environment already has dependencies installed, run:

```bash
python -m pytest -q -m "not slow" --ignore=tests/test_server.py
```

Expected in the current prepared environment: fast non-server tests pass.

Do not claim clean-environment `.[dev]`, Flask server startup, or live Gemini verification
unless those commands are run successfully in the same review session.

---

## Merge Recommendation Rule

Recommend merge only if:

- The diff is documentation-only.
- The stale-claim scan is clean.
- `git diff --check` passes.
- Any runtime checks that were attempted are reported exactly, including failures.

Do not merge until the user explicitly approves after the report.
