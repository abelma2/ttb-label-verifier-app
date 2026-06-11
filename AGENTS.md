# AGENTS.md

This file provides guidance to coding agents (Codex, Cursor, etc.) when working with
code in this repository.

**The canonical, maintained guide is [CLAUDE.md](CLAUDE.md) — read that file in full
before making changes.** It documents the commands, the two-stage architecture (the
vision model reads, deterministic Python judges), the production Next.js + Vercel web
app vs. the legacy Streamlit prototype, the extraction schema contract, and — most
importantly — the regulatory invariants (fail-closed government-warning gate,
class-dependent ABV, brand/class union matching, volume-aware net contents, wine
appellation) that must not be "simplified" away.

This file previously carried its own copy of that guidance; the copy drifted badly out
of date (pre-web-app, missing several verifier invariants) and was replaced by this
pointer so there is a single source of truth. The old content is in git history if you
need it.
