"""
docfinder — a cost-bounded pipeline for finding the governing documents
inside a large document store, without reading everything with an LLM.

Operating point (see DESIGN.md):
  * No human in the loop
  * One-shot batch scan
  * Recall-optimised

The public entry point is `docfinder.pipeline.run`.
"""

from .models import DriveFile
from .pipeline import run, PipelineResult

__all__ = ["DriveFile", "run", "PipelineResult"]
