Full design write-up: [DESIGN.md](DESIGN.md).

Reflection: [REFLECTION.md](REFLECTION.txt).

## The problem

A business connects a Drive of ~50,000 files. About 80 define how it runs. The
rest is noise: invoices, photos, emails, spreadsheets. Running an LLM over all
50,000 costs thousands of pounds. The task is to find the 80 for about the price
of a coffee, with no human checking the result.

A governing document is defined by behaviour, not topic. A policy is authored
internally, revised over years, reviewed by several people, and referenced
elsewhere. An invoice is machine-generated once and never touched. That
difference shows in metadata alone, before reading a word.

## Solution summary

Spend compute inversely proportional to population size: free signals on all
50k, cheap signals on thousands, the LLM on hundreds.

1. Media drop. Hard-drop `image/*`, `video/*`, `audio/*`. The only recall-unsafe
   cut, kept deliberately narrow.
2. Score, do not filter. Every survivor gets one calibrated P(governing) from
   filename, folder, mime, edit-dynamics, a 1-2KB structural peek, and embedding
   similarity to governance anchors. These stages rank, they never drop.
3. Graph rescue. Propagate score along folder and reference edges so a
   weak-metadata governing doc inherits signal from strong neighbours. This is
   what recovers a critical SOP saved as `final_v3.docx` in a junk folder.
4. Verify and audit. Spend ~80% of a fixed LLM budget verifying the ranked head,
   reserve ~20% to sample the near-miss tail. The audit is debiased
   (Rogan-Gladen) against the LLM's own false-positive rate.

Output: a ranked, LLM-verified set plus a measured recall estimate with a
confidence interval. When no human checks the result, a measured number is the
only honest completeness signal.



## Results

One timing run against the Qwen 2.5 3B, using a Intel Core Ultra 7 155H,
16 GB laptop. Compared two ways to find the governing documents in the same
200-file Drive, send every file to the LLM, or let the funnel pre-filter first.

| | Brute-force LLM | Funnel |
|---|---|---|
| LLM calls | 124 | 25 |
| LLM tokens | 27,215 | 5,769 |
| Embedding calls | 0 | 127 |
| Wall-clock time | 534.8s (8.9 min) | 374.6s (6.2 min) |
| Governing found | 6/6 | 6/6 |

The funnel found every governing document in about 1.4x less time, using 5x fewer
LLM calls and tokens. The time gap is modest here because at 200 files the
embedding step adds real overhead on CPU; the decisive numbers are the LLM calls
and tokens. Brute-force sends every file to the LLM, so its calls and tokens
scale with the Drive: 124 calls for 200 files becomes roughly 31,000 for 50,000.
The funnel's LLM work stays fixed at the budget (25 calls) whatever the Drive
size, so the gap widens with scale, and widens further once embeddings are
batched or hosted rather than one CPU call at a time.

Reproduce with `python run_demo.py` (writes a `.txt` and `.json` to `results/`).

## Repository layout

```
docfinder/
  models.py             DriveFile: the pre-content metadata a Drive listing gives you
  corpus.py             synthetic Drive generator (incl. deliberately hard cases)
  signals.py            cheap signals -> one calibrated P(governing)      [Stages 1-3]
  embeddings.py         mock embeddings + governance anchors
  ollama_embeddings.py  real embeddings (nomic-embed-text via Ollama)
  graph.py              folder/reference propagation                      [graph rescue]
  llm.py                mock LLM verifier + cost accounting               [the oracle]
  ollama_llm.py         real verifier (Qwen 2.5 via Ollama)
  pipeline.py           the funnel: rank -> verify head -> audit tail
run_demo.py             CLI: funnel demo, or --benchmark for brute-force vs funnel timing
tests/                  unittest suite, 16 tests, no dependencies
results/                saved reports (.txt + .json)
DESIGN.md               problem framing, architecture, trade-offs
```

## Run the demo

Python 3.9+, no third-party dependencies. By default, `python run_demo.py` runs
the benchmark behind the Results table: brute-force LLM over every file vs the
funnel, on a 200-file Drive. It uses the real local models, so install Ollama and
pull them first:

```bash
winget install Ollama.Ollama        # or https://ollama.com/download
ollama pull qwen2.5:3b              # ~2 GB
ollama pull nomic-embed-text

python run_demo.py                  # ~8 to 9 min on CPU, with a progress bar
```

Prefer the fast offline version? Set `RUN_BENCHMARK = False` in the CONFIG block
at the top of run_demo.py to run the funnel demo on mock models instead
(deterministic, instant, no setup). Files, governing count, budget, seed and
model are all set in that same block, or via the matching command-line flags.

The real backend lives in `ollama_llm.py` and `ollama_embeddings.py`, which keep
the same `verify(f) -> bool` and `anchor_similarity(text)` interfaces as the
mocks, so the funnel is untouched.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

16 tests covering signal ranking, graph rescue including weak-file recovery,
recall-safe media drop, budget enforcement, cost decoupling from corpus size,
and the debiased recall estimate tracking ground truth.
