"""Synthetic corpus generator.

Replicates the real problem at small scale: a store of mostly-noise files with
a small set of governing documents hidden inside. The mix, the metadata
distributions, and the text are chosen to exercise every signal the pipeline
relies on — including deliberately *hard* governing docs (bad names, junk
folders, thin text) that only survive because of the graph-rescue stage.

Everything is seeded, so corpora are reproducible for testing.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import List

from .models import DriveFile

_ORG_USERS = [f"user{i}@acme.co" for i in range(1, 9)]
_EXTERNAL = ["billing@vendor.com", "no-reply@bank.com", "photos@iphone.local"]

_GOV_FOLDERS = ["/HR/Policies", "/Compliance", "/Legal/Contracts",
                "/Operations/SOPs", "/Governance"]
_JUNK_FOLDERS = ["/Downloads", "/Misc", "/2019 stuff", "/Shared/attachments",
                 "/Desktop dump"]

# ---- text templates -------------------------------------------------------

def _policy_text(rng: random.Random, thin: bool = False) -> str:
    if thin:
        # A governing doc with almost no structural signal — the hard case.
        return "Approved by management. See attached.\n"
    secs = "\n".join(f"{i}. Section {i}" for i in range(1, rng.randint(5, 9)))
    return (
        "Table of Contents\n"
        "1. Purpose\n2. Scope\n3. Responsibilities\n4. Procedure\n\n"
        f"Effective Date: 0{rng.randint(1,9)}/2023\n"
        "Version 2.0\n"
        "This document sets out the company policy on the matter herein. "
        "All staff are responsible for compliance with this procedure.\n"
        "Revision History: v1.0, v1.1, v2.0\n"
        f"{secs}\n"
    )


def _contract_text(rng: random.Random) -> str:
    return (
        "AGREEMENT\n"
        "This Agreement is made between the parties as of the Effective Date.\n"
        "1. Term\n2. Obligations\n3. Liability\n4. Termination\n"
        "IN WITNESS WHEREOF the parties have executed this agreement.\n"
        "Signed: ____________\n"
    )


def _invoice_text(rng: random.Random) -> str:
    return (
        f"Invoice Number: INV-{rng.randint(1000,9999)}\n"
        f"Date: 0{rng.randint(1,9)}/2021\n"
        f"Amount Due: £{rng.randint(50, 5000)}.{rng.randint(0,99):02d}\n"
        "Total: as above. Payment due within 30 days.\n"
    )


def _email_text(rng: random.Random) -> str:
    return (
        "From: someone@acme.co\nSubject: quick question\n\n"
        "Hey, can you take a look at this when you get a sec? Thanks!\n"
    )


def _sheet_text(rng: random.Random) -> str:
    rows = "\n".join(",".join(str(rng.randint(0, 999)) for _ in range(5))
                     for _ in range(rng.randint(3, 10)))
    return "col1,col2,col3,col4,col5\n" + rows + "\n"


def _base_times(rng: random.Random):
    created = datetime(2019, 1, 1) + timedelta(days=rng.randint(0, 1500))
    return created


def generate_corpus(
    n_files: int = 3000,
    n_governing: int = 12,
    seed: int = 7,
) -> List[DriveFile]:
    """Generate a corpus of `n_files` with exactly `n_governing` governing docs.

    ~2 of the governing docs are intentionally 'hard' (bad name / junk folder /
    thin text) so the pipeline can only recover them via graph rescue.
    """
    rng = random.Random(seed)
    files: List[DriveFile] = []
    gid = 0

    def new_id() -> str:
        nonlocal gid
        gid += 1
        return f"f{gid:06d}"

    # ---- governing documents ---------------------------------------------
    gov_ids: List[str] = []
    for i in range(n_governing):
        hard = i >= n_governing - 2  # last two are the hard cases
        created = _base_times(rng)
        modified = created + timedelta(days=rng.randint(90, 1200))
        kind = rng.choice(["policy", "sop", "contract"])
        if kind == "contract":
            text = _contract_text(rng)
            folder = "/Legal/Contracts"
            name = rng.choice(["Master Services Agreement.pdf",
                               "Supplier Contract 2023.docx",
                               "NDA - mutual.pdf"])
            mime = "application/pdf"
        else:
            text = _policy_text(rng, thin=hard)
            folder = rng.choice(_GOV_FOLDERS)
            name = rng.choice(["Data Protection Policy.docx",
                               "Health and Safety SOP.docx",
                               "Pricing Policy v2.docx",
                               "Risk Assessment.docx",
                               "Employee Handbook.docx"])
            mime = "application/vnd.google-apps.document"

        if hard:
            # sabotage the cheap signals; only folder co-location will save it
            name = rng.choice(["final_v3.docx", "Untitled document",
                               "doc (2) copy.docx"])
            if rng.random() < 0.5:
                folder = rng.choice(_JUNK_FOLDERS)

        f = DriveFile(
            id=new_id(),
            name=name,
            mime_type=mime,
            size=rng.randint(20_000, 400_000),
            path=folder,
            owner=rng.choice(_ORG_USERS),
            last_modifying_user=rng.choice(_ORG_USERS),
            created_time=created,
            modified_time=modified,
            revision_count=rng.randint(4, 30),      # living document
            shared=True,
            editors=rng.randint(2, 5),
            text=text,
            is_governing=True,
            doc_kind=kind,
        )
        files.append(f)
        gov_ids.append(f.id)

    # ---- noise ------------------------------------------------------------
    n_noise = n_files - n_governing
    for _ in range(n_noise):
        r = rng.random()
        created = _base_times(rng)
        if r < 0.40:  # photos (media, hard-dropped at stage 0)
            f = DriveFile(
                id=new_id(), name=f"IMG_{rng.randint(1000,9999)}.jpg",
                mime_type="image/jpeg", size=rng.randint(1_000_000, 8_000_000),
                path=rng.choice(_JUNK_FOLDERS), owner="photos@iphone.local",
                last_modifying_user="photos@iphone.local",
                created_time=created, modified_time=created,
                revision_count=1, shared=False, text="", doc_kind="photo",
            )
        elif r < 0.70:  # invoices (transactional, templated, never revised)
            f = DriveFile(
                id=new_id(), name=f"invoice_{rng.randint(1000,9999)}.pdf",
                mime_type="application/pdf", size=rng.randint(30_000, 120_000),
                path=rng.choice(["/Finance/Invoices"] + _JUNK_FOLDERS),
                owner="billing@vendor.com",
                last_modifying_user="billing@vendor.com",
                created_time=created, modified_time=created,
                revision_count=1, shared=False,
                text=_invoice_text(rng), doc_kind="invoice",
            )
        elif r < 0.85:  # emails / attachments
            f = DriveFile(
                id=new_id(), name=f"RE_ {rng.choice(['fwd','note','ask'])}.eml",
                mime_type="message/rfc822", size=rng.randint(2_000, 40_000),
                path=rng.choice(_JUNK_FOLDERS), owner=rng.choice(_ORG_USERS),
                last_modifying_user=rng.choice(_ORG_USERS),
                created_time=created, modified_time=created,
                revision_count=1, shared=False,
                text=_email_text(rng), doc_kind="email",
            )
        else:  # spreadsheets / data dumps
            f = DriveFile(
                id=new_id(), name=f"data_{rng.randint(100,999)}.csv",
                mime_type="text/csv", size=rng.randint(5_000, 500_000),
                path=rng.choice(_JUNK_FOLDERS), owner=rng.choice(_ORG_USERS),
                last_modifying_user=rng.choice(_ORG_USERS),
                created_time=created,
                modified_time=created + timedelta(days=rng.randint(0, 30)),
                revision_count=rng.randint(1, 3), shared=rng.random() < 0.2,
                text=_sheet_text(rng), doc_kind="spreadsheet",
            )
        files.append(f)

    # ---- wire up references so graph rescue has something to propagate ----
    # Real governing docs reference each other ("see the Data Protection
    # Policy"). We make ~half the governing docs cite one hard case, giving the
    # hard case an incoming-reference boost.
    hard_ids = gov_ids[-2:]
    for src_id in gov_ids[:-2]:
        if rng.random() < 0.5:
            src = next(f for f in files if f.id == src_id)
            src.references.append(rng.choice(hard_ids))

    rng.shuffle(files)
    return files
