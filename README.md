# AID201A — Classical AI Document Question Answering System (Assignment 1)

A document Question Answering system that accepts a document, builds a symbolic
Knowledge Base (KB) from it, and answers questions using **only** facts in that KB
— with a visible reasoning trace.

**Paradigm:** Classical AI only — knowledge representation (triples + frames),
rule-based inference (forward & backward chaining over Horn clauses), and graph
search (BFS/DFS). **No NLP, no RAG, no embeddings, no vector DB, no LLM.**

## Repository layout
```
kb/          parser, triple store, rules engine, search
qa/          question-template matcher, answerer, trace builder
eval/        gold set (JSON), runner, metrics, ablation
app/         FastAPI backend + vanilla-JS frontend (static)
tests/       pytest suite (parser, rules, 10+ QA cases)
data/        source document + editable rule base
report/      IEEE-format report (LaTeX)
```

## AI techniques used (per module)
- `kb/parser.py` — Knowledge Representation via structure-driven extraction
  (headings, tables, `key: value` lines, deterministic clause templates).
- `kb/store.py` — TripleStore: indexed (s,p,o) KB + BFS/DFS graph search.
- `kb/rules.py` — Horn-clause Rule Engine: forward chaining + backward chaining
  with numeric comparisons (`cmp:ge/lt/eq`).
- `qa/answerer.py` — Template-based question parsing + lookup/inference +
  inference-chain trace builder (fact → rule → derived fact → answer).
- `eval/` — Gold set (30+ Q&A, ≥10 inference, ≥5 unanswerable), accuracy /
  precision / recall / abstention, per-category breakdown, ablation (with vs
  without rules).
- `app/` — FastAPI backend + polished vanilla-JS frontend (3-panel desktop,
  bottom tab bar mobile, light/dark theme).

## Quick start
```bash
# 1. (Optional) create a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
python run.py
# UI: http://127.0.0.1:8000   |   API docs: http://127.0.0.1:8000/docs
```

## Run the test suite
```bash
python -m pytest tests/ -v
```

## Run the evaluation (gold set + ablation)
The evaluation dashboard is available from the UI ("Evaluation & Ablation").
Programmatically:
```bash
python -c "
from eval.runner import run_suite
m = run_suite(with_rules=True)['metrics']
print(m)
"
```

## Requirements
Python 3.10+. See `requirements.txt`.

## Constraints honoured
- No spaCy / NLTK / transformers / embeddings / FAISS / Chroma / LangChain / LLM for
  answering. Facts are extracted deterministically from document structure only.
- Answers are never guessed: unsupported questions return "Not in document"
  (confidence 0) with an explicit abstention trace.
