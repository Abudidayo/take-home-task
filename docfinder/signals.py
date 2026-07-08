"""Cheap signals — everything computable without an LLM.

Each function returns a feature in [0, 1]. They are combined into a single
calibrated-ish probability by `score_file`. In production these weights would
be *learned* from cross-business labels (see DESIGN.md §"Getting smarter");
here they are hand-set so the prototype is self-contained and deterministic.

Design rule: none of these signals ever *drops* a file. They only rank it.
The only hard drop happens in the pipeline's stage 0 (media).

============================================================================
WHAT THE APPLICATION LOOKS FOR (all pre-LLM, all case-insensitive)
============================================================================
The lists below are the *entire* vocabulary the ranker uses before any LLM is
called. They are defined as plain tuples/dicts so they are easy to read, audit
and extend. What the real LLM looks for is a separate, higher-level definition:
see SYSTEM_PROMPT in ollama_llm.py.
"""

from __future__ import annotations

import math
import re

from .models import DriveFile

# (1) FILENAME keywords -----------------------------------------------------
# A filename containing any of these is a strong positive prior: governing
# documents are usually named for what they are ("Data Protection Policy.docx").
NAME_KEYWORDS = (
    "policy", "policies", "sop", "procedure", "procedures", "handbook",
    "manual", "agreement", "contract", "nda", "risk assessment", "terms",
    "guidelines", "charter", "framework", "code of conduct",
)

# (2) FOLDER-path keywords --------------------------------------------------
# A file whose folder path contains any of these inherits a positive prior;
# businesses file governance documents in named folders ("/HR/Policies").
FOLDER_KEYWORDS = (
    "policies", "policy", "compliance", "legal", "contracts", "governance",
    "sops", "sop", "hr", "regulatory",
)

# (3) MIME-type priors ------------------------------------------------------
# Governing docs are word-processor documents or PDFs; data and mail rank low.
# (Media types never reach here: they are hard-dropped in pipeline stage 0.)
# value = prior weight in [0, 1].
_GOV_MIME = {
    "application/vnd.google-apps.document": 1.0,   # Google Doc
    "application/pdf": 0.8,                         # PDF
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": 1.0,  # .docx
    "application/msword": 1.0,                      # legacy .doc
    "application/vnd.google-apps.spreadsheet": 0.4,  # Google Sheet
    "text/csv": 0.2,                               # CSV data dump
    "message/rfc822": 0.1,                         # email
}

# (4) STRUCTURAL markers in the first 1-2KB of text -------------------------
# The skeleton of a governing document. A policy or contract has these; an
# invoice or photo does not.
STRUCTURE_MARKERS = (
    r"table of contents",
    r"\bpurpose\b",
    r"\bscope\b",
    r"responsibilit",               # responsibility / responsibilities
    r"effective date",
    r"\bversion\b|\bv\d+\.\d+\b",    # "Version 2.0" / "v2.0"
    r"revision history",
    r"^\s*\d+\.\s",                 # numbered sections: "1. ..."
    r"in witness whereof|signed:",  # contract execution / signature blocks
)

# (5) ANTI-markers ----------------------------------------------------------
# Text that argues *against* a governing document, penalising the score.
ANTI_MARKERS = (
    r"invoice number|amount due|payment due",   # invoices, receipts
    r"^from:.*subject:",                         # email headers
)

# (6) METADATA fields read by edit_dynamics_signal --------------------------
# The "living document" fingerprint uses these DriveFile fields (see models.py):
#   revision_count, created_time, modified_time, editors, shared.

# ---- compiled forms (behaviour identical to the tuples above) -------------
_NAME_KEYWORDS = re.compile(r"\b(" + "|".join(NAME_KEYWORDS) + r")\b", re.IGNORECASE)
_FOLDER_KEYWORDS = re.compile("|".join(FOLDER_KEYWORDS), re.IGNORECASE)
_STRUCT_PATTERNS = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in STRUCTURE_MARKERS]
_ANTI_PATTERNS = [re.compile(p, re.IGNORECASE | re.MULTILINE | re.DOTALL) for p in ANTI_MARKERS]


def name_signal(f: DriveFile) -> float:
    return 1.0 if _NAME_KEYWORDS.search(f.name) else 0.0


def folder_signal(f: DriveFile) -> float:
    return 1.0 if _FOLDER_KEYWORDS.search(f.path) else 0.0


def mime_signal(f: DriveFile) -> float:
    return _GOV_MIME.get(f.mime_type, 0.0)


def edit_dynamics_signal(f: DriveFile) -> float:
    """The 'living document' fingerprint: revised over time, multiple editors,
    shared. An invoice scores ~0; a policy scores high."""
    rev = min(f.revision_count / 10.0, 1.0)          # many revisions
    span = min(f.age_days / 365.0, 1.0)              # revised over a long span
    people = min((f.editors - 1) / 3.0, 1.0)        # multiple editors
    shared = 1.0 if f.shared else 0.0
    return 0.4 * rev + 0.3 * span + 0.2 * people + 0.1 * shared


def structure_signal(f: DriveFile) -> float:
    """Structural fingerprint from the first ~1-2KB of text (a cheap peek)."""
    if not f.text:
        return 0.0
    head = f.text[:2000]
    hits = sum(1 for p in _STRUCT_PATTERNS if p.search(head))
    anti = sum(1 for p in _ANTI_PATTERNS if p.search(head))
    raw = hits - 1.5 * anti
    return max(0.0, min(raw / 4.0, 1.0))


# ---- combined score -------------------------------------------------------

# Hand-set weights (a stand-in for a trained logistic regression). Order:
# name, folder, mime, edit_dynamics, structure, embedding.
_WEIGHTS = {
    "bias": -3.4,
    "name": 2.2,
    "folder": 1.8,
    "mime": 1.0,
    "edit": 1.6,
    "structure": 2.4,
    "embedding": 2.0,
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def score_file(f: DriveFile, embedding_sim: float) -> float:
    """Return calibrated-ish P(governing) from cheap signals + embedding sim.

    `embedding_sim` is supplied by the caller (see embeddings.py) so this stays
    pure and easy to test.
    """
    logit = (
        _WEIGHTS["bias"]
        + _WEIGHTS["name"] * name_signal(f)
        + _WEIGHTS["folder"] * folder_signal(f)
        + _WEIGHTS["mime"] * mime_signal(f)
        + _WEIGHTS["edit"] * edit_dynamics_signal(f)
        + _WEIGHTS["structure"] * structure_signal(f)
        + _WEIGHTS["embedding"] * embedding_sim
    )
    return _sigmoid(logit)
