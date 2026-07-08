"""Core data model.

A `DriveFile` is the small subset of Google Drive metadata that is available
*before* we read a file's contents, plus (optionally) the extractable text.

`is_governing` is the ground-truth label. In production this does not exist;
here it is used only to (a) generate a realistic synthetic corpus and
(b) evaluate the pipeline. The pipeline itself never reads this field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List


# MIME types we treat as "definitely not a governing document" and hard-drop
# at stage 0. This is the ONLY recall-unsafe operation in the pipeline, so the
# set is deliberately conservative: a JPEG cannot be a policy.
MEDIA_MIME_PREFIXES = ("image/", "video/", "audio/")


@dataclass
class DriveFile:
    id: str
    name: str
    mime_type: str
    size: int                       # bytes
    path: str                       # containing folder, e.g. "/HR/Policies"
    owner: str                      # email
    last_modifying_user: str        # email
    created_time: datetime
    modified_time: datetime
    revision_count: int             # number of stored revisions
    shared: bool
    editors: int = 1                # distinct users who have edited it
    text: str = ""                  # extractable text (empty for media/scanned)
    references: List[str] = field(default_factory=list)  # ids this file links to

    # ---- ground truth: NEVER read by the pipeline, only by corpus/eval ----
    is_governing: bool = False
    doc_kind: str = "other"         # policy | contract | sop | invoice | photo | ...

    @property
    def is_media(self) -> bool:
        return self.mime_type.startswith(MEDIA_MIME_PREFIXES)

    @property
    def age_days(self) -> float:
        return (self.modified_time - self.created_time).total_seconds() / 86400.0
