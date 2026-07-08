"""Real local-LLM verifier, backed by Ollama running Qwen 2.5.

Drop-in replacement for the mock `LLMVerifier`: same `verify(f) -> bool`
interface and the same `.calls` / `.tokens` accounting, so the funnel in
`pipeline.py` does not change. Selected via `run_demo.py --backend ollama`.

Uses only the standard library (urllib) so the project stays dependency-free.
Requires a running Ollama service (`ollama serve`) with the model pulled:

    ollama pull qwen2.5:3b
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List

from .models import DriveFile

# What the LLM looks for. A crisp, explicit functional definition is what makes
# a small model reliable at this binary decision. The GOVERNING and OTHER lists
# below are the exact document types the model is told to recognise; they mirror
# the filename keywords the application scores on (NAME_KEYWORDS in signals.py).
SYSTEM_PROMPT = (
    "You classify files from a company Google Drive. Reply with exactly one "
    "word: GOVERNING or OTHER.\n"
    "GOVERNING documents define how the business runs. Examples: policy, "
    "procedure, SOP (standard operating procedure), contract, agreement, NDA, "
    "risk assessment, employee handbook, staff manual, code of conduct, terms "
    "and conditions, guidelines, charter, framework.\n"
    "OTHER is everything else. Examples: invoices, receipts, purchase orders, "
    "photos, casual emails, meeting notes, raw data or spreadsheets, marketing "
    "material, personal files.\n"
    "Judge by the document's function, not its topic."
)

# The LLM is shown only three fields per file (see `_prompt`): the filename, the
# folder path, and the first ~1500 characters of text. It never sees the whole
# document, which is what keeps the expensive stage cheap.


@dataclass
class OllamaVerifier:
    model: str = "qwen2.5:3b"
    host: str = "http://localhost:11434"
    timeout: float = 60.0
    first_page_chars: int = 1500        # only the cheap peek reaches the LLM
    calls: int = 0
    tokens: int = 0
    _log: List[str] = field(default_factory=list)

    def _prompt(self, f: DriveFile) -> str:
        return (
            f"Filename: {f.name}\n"
            f"Folder: {f.path}\n"
            f"First page: {f.text[:self.first_page_chars]}"
        )

    def verify(self, f: DriveFile) -> bool:
        """Ask Qwen whether `f` is a governing document. Returns True/False.

        On any transport/parse error we return False (conservative: do not
        surface junk) and still count the call so cost accounting is honest.
        """
        self.calls += 1
        self._log.append(f.id)
        payload = json.dumps({
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": self._prompt(f),
            "stream": False,
            "options": {"temperature": 0, "num_predict": 3},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as e:  # pragma: no cover
            print(f"  [ollama] call failed for {f.id}: {e}")
            return False

        self.tokens += int(data.get("prompt_eval_count", 0)) + int(data.get("eval_count", 0))
        answer = (data.get("response") or "").strip().upper()
        return answer.startswith("GOVERNING")

    @staticmethod
    def naive_token_cost(files: List[DriveFile]) -> int:
        """Estimated tokens the baseline would spend reading every text file
        in full — the yardstick the funnel is measured against."""
        return sum(min(len(f.text), 6000) // 4 + 20 for f in files if f.text)
