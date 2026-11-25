"""FastAPI Backend for AID201A Classical AI Document QA System.

Features:
  - Structure-driven document parsing (Markdown, TXT, DOCX, PDF)
  - Pure symbolic TripleStore Knowledge Base
  - Horn-clause Rule Engine (Forward & Backward chaining)
  - Deterministic Question Answering with visible reasoning trace
  - Live Evaluation & Ablation endpoints
  - Single-command execution serving both API and UI
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from eval.evaluator import run_ablation
from kb.parser import _split_sections, parse_file, parse_text
from kb.rules import load_rules_from_json
from kb.store import TripleStore
from qa.answerer import Answerer
from qa.templates import template_names

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    if DEFAULT_DOC.exists() and STATE["store"].fact_count() == 0:
        _load_document(DEFAULT_DOC, "handbook.md")
    yield


app = FastAPI(
    title="AID201A Classical AI Document Question Answering System",
    description="Knowledge representation, rule-based inference, and search over document triples.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State
WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DOC = WORKSPACE_DIR / "data" / "handbook.md"
DEFAULT_RULES = WORKSPACE_DIR / "data" / "rules.json"

STATE: dict[str, Any] = {
    "store": TripleStore(),
    "doc_name": "handbook.md",
    "doc_text": "",
    "sections": [],
    "rules_path": str(DEFAULT_RULES),
}


def _load_document(doc_path: Path, doc_name: str | None = None) -> None:
    text = doc_path.read_text(encoding="utf-8", errors="replace")
    facts = parse_file(doc_path)
    store = TripleStore()
    store.add_facts(facts)

    if DEFAULT_RULES.exists():
        rules = load_rules_from_json(DEFAULT_RULES)
        store.add_rules(rules)

    lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
    sections = _split_sections(lines)

    STATE["store"] = store
    STATE["doc_name"] = doc_name or doc_path.name
    STATE["doc_text"] = text
    STATE["sections"] = [
        {
            "heading": s.heading,
            "line_count": len(s.lines),
            "preview": "\n".join(s.lines[:6]),
            "full_text": "\n".join(s.lines),
        }
        for s in sections
    ]


# Initialize with default sample document on module load
if DEFAULT_DOC.exists():
    _load_document(DEFAULT_DOC, "handbook.md")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class QuestionRequest(BaseModel):
    question: str


class RuleToggleRequest(BaseModel):
    rule_name: str
    enabled: bool


# --------------------------------------------------------------------------- #
# Document & Knowledge Base Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict[str, Any]:
    store: TripleStore = STATE["store"]
    return {
        "status": "healthy",
        "doc_name": STATE["doc_name"],
        "fact_count": store.fact_count(),
        "rule_count": store.rule_count(),
        "section_count": len(STATE["sections"]),
    }


@app.get("/api/kb/stats")
def kb_stats() -> dict[str, Any]:
    store: TripleStore = STATE["store"]
    return {
        "doc_name": STATE["doc_name"],
        "facts": store.fact_count(),
        "rules": store.rule_count(),
        "sections": len(STATE["sections"]),
        "predicates": sorted({f.triple.predicate for f in store.facts()}),
        "subjects_count": len({f.triple.subject for f in store.facts()}),
    }


@app.get("/api/kb/schema")
def kb_schema() -> dict[str, Any]:
    return {
        "architecture": "Classical Symbolic AI (No NLP, No LLM, No Vector DB)",
        "components": {
            "Triple": {"fields": ["subject", "predicate", "object"], "description": "Atomic unit of knowledge"},
            "Fact": {"fields": ["triple", "provenance", "confidence"], "description": "Stored triple linked to section/clause provenance"},
            "Provenance": {"fields": ["section", "clause", "page"], "description": "Document source citation"},
            "Rule": {"fields": ["name", "head", "body", "provenance", "enabled"], "description": "Horn clause for forward/backward chaining"},
            "QueryTemplates": template_names(),
        },
    }


@app.get("/api/kb/facts")
def kb_facts(q: str = Query("", description="Search term for subject, predicate, or object")) -> dict[str, Any]:
    store: TripleStore = STATE["store"]
    facts = store.facts()
    if q.strip():
        term = q.strip().lower()
        facts = [
            f for f in facts
            if term in f.triple.subject.lower()
            or term in f.triple.predicate.lower()
            or term in f.triple.object.lower()
            or term in f.provenance.section.lower()
        ]

    return {
        "total": len(facts),
        "facts": [
            {
                "subject": f.triple.subject,
                "predicate": f.triple.predicate,
                "object": f.triple.object,
                "provenance": f.provenance.to_dict(),
                "citation": f.provenance.short(),
            }
            for f in facts
        ],
    }


@app.get("/api/kb/rules")
def kb_rules() -> dict[str, Any]:
    store: TripleStore = STATE["store"]
    return {
        "total": store.rule_count(),
        "rules": [
            {
                "name": r.name,
                "head": r.head.to_dict(),
                "body": [c.to_dict() for c in r.body],
                "provenance": r.provenance.to_dict() if r.provenance else None,
                "enabled": r.enabled,
            }
            for r in store.rules()
        ],
    }


@app.post("/api/kb/rules/toggle")
def toggle_rule(req: RuleToggleRequest) -> dict[str, Any]:
    store: TripleStore = STATE["store"]
    found = False
    for r in store.rules():
        if r.name == req.rule_name:
            r.enabled = req.enabled
            found = True
            break
    if not found:
        raise HTTPException(status_code=404, detail=f"Rule '{req.rule_name}' not found")
    return {"status": "ok", "rule_name": req.rule_name, "enabled": req.enabled}


@app.get("/api/kb/sections")
def kb_sections() -> dict[str, Any]:
    return {
        "doc_name": STATE["doc_name"],
        "total": len(STATE["sections"]),
        "sections": STATE["sections"],
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or "uploaded_document"
    ext = Path(filename).suffix.lower()
    if ext not in (".md", ".txt", ".docx", ".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: .md, .txt, .docx, .pdf",
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = Path(tmp.name)
        shutil.copyfileobj(file.file, tmp)

    try:
        _load_document(tmp_path, filename)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

    store: TripleStore = STATE["store"]
    return {
        "status": "success",
        "doc_name": filename,
        "facts_count": store.fact_count(),
        "rules_count": store.rule_count(),
        "sections_count": len(STATE["sections"]),
        "summary": f"{store.fact_count()} facts · {store.rule_count()} rules · {len(STATE['sections'])} sections",
    }


@app.post("/api/load-sample")
def load_sample() -> dict[str, Any]:
    if not DEFAULT_DOC.exists():
        raise HTTPException(status_code=404, detail="Default handbook.md not found")
    _load_document(DEFAULT_DOC, "handbook.md")
    store: TripleStore = STATE["store"]
    return {
        "status": "success",
        "doc_name": "handbook.md",
        "facts_count": store.fact_count(),
        "rules_count": store.rule_count(),
        "sections_count": len(STATE["sections"]),
        "summary": f"{store.fact_count()} facts · {store.rule_count()} rules · {len(STATE['sections'])} sections",
    }


# --------------------------------------------------------------------------- #
# Question Answering Endpoint
# --------------------------------------------------------------------------- #
@app.post("/api/qa")
def answer_question(req: QuestionRequest) -> dict[str, Any]:
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    store: TripleStore = STATE["store"]
    answerer = Answerer(store)
    ans = answerer.answer(q)
    return ans.to_dict()


# --------------------------------------------------------------------------- #
# Evaluation Endpoint
# --------------------------------------------------------------------------- #
@app.get("/api/eval")
@app.post("/api/eval/run")
def run_evaluation() -> dict[str, Any]:
    store: TripleStore = STATE["store"]
    test_path = WORKSPACE_DIR / "eval" / "gold_test_set.json"
    rules_p = STATE.get("rules_path", str(DEFAULT_RULES))
    report = run_ablation(
        doc_path=DEFAULT_DOC,
        rules_path=rules_p,
        test_set_path=test_path,
    )
    return report


# --------------------------------------------------------------------------- #
# Static Files & Frontend Mount
# --------------------------------------------------------------------------- #
FRONTEND_DIR = WORKSPACE_DIR / "app" / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"message": "Frontend index.html not found"})
