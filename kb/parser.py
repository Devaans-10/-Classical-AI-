"""Document parser: builds the Knowledge Base from a source document.

AI technique: Knowledge Representation via structure-driven extraction.

Why no NLP/RAG: The assignment forbids language understanding for fact
extraction. Instead we extract facts deterministically from *structure*:
headings (sections), tables (component/weight/marks rows), numbered clauses
("Clause 2.1.1: ..."), and "key: value" lines. Every extracted fact carries
provenance (section, clause, page) so answers are always citable and never
guessed.

Supported input formats:
  * Markdown (.md)    - native, full feature set (primary path).
  * Plain text (.txt)  - key: value lines and clause lines.
  * DOCX (.docx)       - pure-Python (zipfile + xml.etree) & python-docx fallback.
  * PDF (.pdf)         - pypdf if available, else pure-Python flate/stream decoder.

Parsing pipeline:
  1. Read raw text (format-specific).
  2. Split into sections by heading lines.
  3. Within each section, extract triples from tables, then key:value lines,
     then numbered clause sentences, using deterministic templates authored in
     this module.
  4. Return a list of Fact objects for the TripleStore.
"""

from __future__ import annotations

import io
import re
import zipfile
import zlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kb.types import Fact, Provenance, Triple, _norm_subject


@dataclass
class Section:
    """A section of the document: heading + body lines + optional table rows."""

    heading: str
    lines: list[str]
    tables: list[list[list[str]]] = None  # list of tables; each table = list of rows (cells)

    def __post_init__(self) -> None:
        if self.tables is None:
            self.tables = []


# --------------------------------------------------------------------------- #
# Template / clause sentence patterns (authored deterministically)
# --------------------------------------------------------------------------- #
_CLAUSE_RE = re.compile(
    r"^\s*Clause\s+(?P<clause>[\d]+(?:\.[\d]+)*)\s*[:]\s*(?P<text>.+)$",
    re.IGNORECASE,
)
_CLAUSE_RE_NO_COLON = re.compile(
    r"^\s*Clause\s+(?P<clause>[\d]+(?:\.[\d]+)*)\s+(?P<text>.+)$",
    re.IGNORECASE,
)

# "key: value" line
_KV_RE = re.compile(r"^\s*(?P<key>[^:]{2,60}):\s*(?P<value>.+?)\s*$")

# Numeric fact sentence: "X is Y" / "X: Y" handled by KV; percentage sentences:
_PERCENT_RE = re.compile(
    r"(?P<subj>[^.]*?)\s+(?P<op>greater than or equal to|greater than|less than|below|above|>=|<=|>|<)\s*(?P<val>[\d.]+)\s*(?P<unit>percent|%|days|hours|INR|INR|kg)?",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


_MD_STRIP_RE = re.compile(r"[\*_`~]{1,3}")


def _strip_md(text: str) -> str:
    """Remove inline Markdown emphasis so keys/values are clean."""
    return _MD_STRIP_RE.sub("", text).strip()


# --------------------------------------------------------------------------- #
# Section splitting
# --------------------------------------------------------------------------- #
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(?P<title>.+?)\s*#*\s*$")


def _split_sections(lines: list[str]) -> list[Section]:
    """Split document lines into sections by Markdown headings.

    Lines before the first heading belong to an implicit "Preamble" section.
    """
    sections: list[Section] = []
    current: Section | None = None
    preamble_lines: list[str] = []

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            if current is not None:
                sections.append(current)
            current = Section(heading=_clean(m.group("title")), lines=[])
        else:
            if current is None:
                preamble_lines.append(line)
            else:
                current.lines.append(line)

    if preamble_lines:
        sections.insert(0, Section(heading="Preamble", lines=preamble_lines))
    if current is not None:
        sections.append(current)
    return sections


# --------------------------------------------------------------------------- #
# Table extraction (Markdown)
# --------------------------------------------------------------------------- #
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\s*\|?\s*$")
_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")


def _parse_markdown_tables(lines: list[str]) -> list[list[list[str]]]:
    """Extract Markdown tables from a list of lines.

    Supports both tables with divider rows (|---|---|) and tables without
    divider rows. Returns list of tables, each table is a list of rows.
    """
    tables: list[list[list[str]]] = []
    i = 0
    n = len(lines)
    while i < n:
        # Check if current line is a table row
        if _ROW_RE.match(lines[i]):
            rows: list[list[str]] = []
            while i < n and _ROW_RE.match(lines[i]):
                line = lines[i]
                if _TABLE_SEP_RE.match(line):
                    # separator row - skip it
                    i += 1
                    continue
                cells = [c.strip() for c in _ROW_RE.match(line).group(1).split("|")]
                if any(cells):
                    rows.append(cells)
                i += 1
            if len(rows) >= 2:
                tables.append(rows)
        else:
            i += 1
    return tables


def _table_to_triples(table: list[list[str]], section: str,
                      provenance: Provenance) -> list[Fact]:
    """Convert a Markdown table to triples using its first column as subject.

    Assumes the table's first column names the entity/component and the
    remaining columns are attributes. Row i produces triples
    (row[0], header[j], row[j]) for j >= 1.
    """
    facts: list[Fact] = []
    if len(table) < 2:
        return facts
    header = table[0]
    for row in table[1:]:
        if not row or not row[0]:
            continue
        subject = _norm_subject(row[0])
        for j in range(1, len(header)):
            if j >= len(row):
                continue
            pred = _strip_md(_clean(header[j])).lower()
            obj = _strip_md(_clean(row[j]))
            if not obj:
                continue
            facts.append(Fact(Triple(subject, pred, obj), provenance))
    return facts


# --------------------------------------------------------------------------- #
# Clause / sentence extraction
# --------------------------------------------------------------------------- #
def _extract_from_clause_line(line: str, section: str,
                              provenance: Provenance) -> list[Fact]:
    """Turn a numbered clause sentence into one or more triples."""
    facts: list[Fact] = []
    m = _CLAUSE_RE.match(line) or _CLAUSE_RE_NO_COLON.match(line)
    if not m:
        return facts
    clause = m.group("clause")
    text = _clean(m.group("text"))
    prov = Provenance(section=section, clause=f"Clause {clause}")

    # Canonical retrievable fact: the clause statement itself.
    facts.append(Fact(Triple(section, f"clause_{clause}", text), prov))

    low = text.lower()

    # Deterministic linguistic templates (authored, not learned).
    templates = [
        # "A student with attendance >= 75 percent is eligible to appear ..."
        (r"attendance\s+(greater than or equal to|>=|>)\s*([\d.]+)\s*percent.*"
         r"eligible to appear",
         lambda mm: [Fact(Triple("a student", "eligible_for_exam_if_attendance_ge", mm.group(2)), prov)]),
        # "attendance below 60 percent is debarred"
        (r"attendance\s+(below|<|less than)\s*([\d.]+)\s*percent.*debarred",
         lambda mm: [Fact(Triple("a student", "debarred_if_attendance_lt", mm.group(2)), prov)]),
        # "medical leave up to N days is condoned"
        (r"medical leave up to ([\d.]+)\s*days?.*condoned",
         lambda mm: [Fact(Triple("a student", "medical_leave_condoned_max_days", mm.group(1)), prov)]),
        # "attendance between 60 percent and 74 percent requires ... fee of N INR"
        (r"attendance between ([\d.]+)\s*percent and ([\d.]+)\s*percent requires.*fee of ([\d.]+)\s*(inr)?",
         lambda mm: [Fact(Triple("a student", "condonation_fee_inr", mm.group(3)), prov),
                     Fact(Triple("attendance policy", "condonation_fee_inr", mm.group(3)), prov)]),
        # "A candidate found using unfair means is debarred for one semester"
        (r"unfair means.*debarred for ([\d.]+)\s*semester",
         lambda mm: [Fact(Triple("unfair means", "debarred_for_semesters", mm.group(1)), prov)]),
        # "revaluation must be applied for within N days"
        (r"revaluation.*within ([\d.]+)\s*days",
         lambda mm: [Fact(Triple("revaluation", "apply_within_days", mm.group(1)), prov)]),
        # "books issued per student: N"
        (r"books issued per student.*?([\d.]+)",
         lambda mm: [Fact(Triple("library", "books_issued_per_student", mm.group(1)), prov)]),
        # "loan duration: N days"
        (r"loan duration.*?([\d.]+)\s*days",
         lambda mm: [Fact(Triple("library", "loan_duration_days", mm.group(1)), prov)]),
        # "overdue fine ... N ... per day"
        (r"overdue fine.*?([\d.]+).*per day",
         lambda mm: [Fact(Triple("library", "overdue_fine_per_day_per_book_inr", mm.group(1)), prov)]),
        # "curfew time: HH:MM"
        (r"curfew time.*?(\d{1,2}:\d{2})",
         lambda mm: [Fact(Triple("hostel", "curfew_time", mm.group(1)), prov)]),
    ]

    for pattern, factory in templates:
        mm = re.search(pattern, low)
        if mm:
            try:
                facts.extend(factory(mm))
            except Exception:
                pass
            break
    return facts


def _extract_from_kv_line(line: str, section: str,
                          provenance: Provenance) -> list[Fact]:
    """Extract a triple from a 'key: value' line."""
    m = _KV_RE.match(line)
    if not m:
        return []
    key = _strip_md(_clean(m.group("key")))
    value = _strip_md(_clean(m.group("value")))
    if not key or not value:
        return []
    subject = _norm_subject(section)
    predicate = key.lower()
    return [Fact(Triple(subject, predicate, value), provenance)]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def parse_text(text: str) -> list[Fact]:
    """Parse raw document text into a list of Fact objects.

    Pipeline: split into sections -> for each section extract table triples,
    then key:value line triples, then numbered-clause triples.
    """
    lines = [_strip_md(ln.rstrip("\n")) for ln in text.splitlines()]
    sections = _split_sections(lines)
    facts: list[Fact] = []

    for section in sections:
        prov = Provenance(section=section.heading)

        # 1. Tables first (most structured).
        for table in _parse_markdown_tables(section.lines):
            facts.extend(_table_to_triples(table, section.heading, prov))

        # 2. key: value lines.
        for line in section.lines:
            if _TABLE_SEP_RE.match(line) or _ROW_RE.match(line):
                continue
            if _CLAUSE_RE.match(line) or _CLAUSE_RE_NO_COLON.match(line):
                continue
            facts.extend(_extract_from_kv_line(line, section.heading, prov))

        # 3. Numbered clauses.
        for line in section.lines:
            if _CLAUSE_RE.match(line) or _CLAUSE_RE_NO_COLON.match(line):
                facts.extend(_extract_from_clause_line(line, section.heading,
                                                       Provenance(section=section.heading)))

    return facts


def parse_file(path: str | Path) -> list[Fact]:
    """Parse a file into facts, dispatching by extension.

    Supports Markdown (.md), Plain Text (.txt), Word (.docx), and PDF (.pdf).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".md", ".txt"):
        text = path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".docx":
        text = _extract_docx_text(path)
    elif suffix == ".pdf":
        text = _extract_pdf_text(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    return parse_text(text)


def _extract_docx_text(path: Path) -> str:
    """Extract text and tables from a DOCX file using pure Python."""
    # 1. Try python-docx if installed
    try:
        import docx  # type: ignore
        d = docx.Document(str(path))
        lines: list[str] = []
        for p in d.paragraphs:
            if p.text.strip():
                lines.append(p.text.strip())
        for table in d.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)
    except Exception:
        pass

    # 2. Pure-Python fallback using zipfile + xml.etree
    try:
        with zipfile.ZipFile(path, "r") as zf:
            if "word/document.xml" not in zf.namelist():
                return path.read_text(encoding="utf-8", errors="replace")
            xml_content = zf.read("word/document.xml")
            root = ET.fromstring(xml_content)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

            lines: list[str] = []
            body = root.find("w:body", ns)
            if body is not None:
                for elem in body:
                    tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                    if tag == "p":
                        # Paragraph
                        texts = [t.text for t in elem.iterfind(".//w:t", ns) if t.text]
                        p_text = "".join(texts).strip()
                        if p_text:
                            # Check if heading
                            pPr = elem.find("w:pPr", ns)
                            if pPr is not None:
                                pStyle = pPr.find("w:pStyle", ns)
                                if pStyle is not None and "val" in pStyle.attrib:
                                    s_val = pStyle.attrib["val"].lower()
                                    if "heading" in s_val:
                                        p_text = f"## {p_text}"
                            lines.append(p_text)
                    elif tag == "tbl":
                        # Table
                        for tr in elem.iterfind(".//w:tr", ns):
                            cells: list[str] = []
                            for tc in tr.iterfind(".//w:tc", ns):
                                cell_texts = [t.text for t in tc.iterfind(".//w:t", ns) if t.text]
                                cells.append("".join(cell_texts).strip())
                            if any(cells):
                                lines.append("| " + " | ".join(cells) + " |")
            return "\n".join(lines)
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace")


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF file using pypdf or pure Python streams."""
    # 1. Try pypdf if installed
    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        pass

    # 2. Pure-Python stream decoder
    try:
        data = path.read_bytes()
        text_parts: list[str] = []

        # Find FlateDecode streams
        stream_re = re.compile(b"stream[\r\n]+(.*?)[\r\n]+endstream", re.DOTALL)
        for m in stream_re.finditer(data):
            stream_data = m.group(1)
            try:
                decomp = zlib.decompress(stream_data)
                # Find strings in parentheses inside BT ... ET text blocks
                # e.g., (Text) Tj or [(T) -10 (ext)] TJ
                for tj_match in re.finditer(rb"\(([^)]+)\)\s*Tj", decomp):
                    decoded = tj_match.group(1).decode("latin-1", errors="replace")
                    if decoded.strip():
                        text_parts.append(decoded)
                for array_match in re.finditer(rb"\[(.*?)\]\s*TJ", decomp):
                    for piece in re.finditer(rb"\(([^)]+)\)", array_match.group(1)):
                        text_parts.append(piece.group(1).decode("latin-1", errors="replace"))
            except Exception:
                continue

        if text_parts:
            return "\n".join(text_parts)

        # Fallback to ASCII string extraction
        strings = re.findall(rb"[\x20-\x7E\r\n]{4,}", data)
        return "\n".join(s.decode("latin-1", errors="replace") for s in strings)
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace")


__all__ = ["parse_text", "parse_file", "Section", "Fact"]
