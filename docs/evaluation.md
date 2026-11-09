# Evaluation

How the system's quality is measured, and how the numbers behind its settings are chosen. Code: `evals/`.

## Why evaluate

Every number in this project started as an educated default: 400-token chunks, 5 chunks per question, a 0.8 distance cut-off, `gpt-6-luna` for answers. Defaults are guesses until measured. The evaluation suite asks the same fixed questions against the same fixed documents, so any change (a setting, a prompt, a model) can be compared fairly with what came before.

## The test set

**Documents.** Nine pages from two openly licensed company handbooks: PostHog (MIT) and Sourcegraph (Apache 2.0), about 9,400 words. Both companies write about similar topics (time off, learning, equity), which tests whether the system keeps one company's policy apart from the other's, a common way RAG systems fail. Sources, licences and the small cleanups made are listed in `evals/documents/licences/NOTICE.md`; `python -m evals.prepare_documents` re-creates the documents exactly.

**Questions.** 60 questions in `evals/dataset/questions.jsonl`, each with a reference answer written by a person:

| Type | Count | Tests |
|---|---|---|
| `fact` | 24 | A single stated fact |
| `number` | 18 | A specific figure, where a near miss is a wrong answer |
| `table` | 4 | Reading a row from a table |
| `multi_part` | 6 | An answer made of several facts, sometimes in different places |
| `comparison` | 2 | Facts from both companies in one answer |
| `unanswerable` | 6 | Not in the documents; the right answer is "not covered" |

Several unanswerable questions are traps: "How many days off does *Sourcegraph* give?" has an answer for PostHog, and a system that mixes the two up fails.

Each answerable question also lists **evidence**: short phrases copied word for word from the document that prove the answer. A test checks that every phrase really is in its document.

## What is measured

### Retrieval (no chat model needed)

For each chunk size (200, 400 and 800 tokens), the documents are ingested and every question is searched:

- **Hit@k**: share of questions whose evidence is all within the top *k* chunks. Checked by looking for the evidence phrases, so no model judges this.
- **MRR** (mean reciprocal rank): 1.0 if the evidence is always the first chunk, 0.5 if typically second, and so on.
- **Distance cut-off table**: for each possible cut-off, how many questions keep their evidence, and how many unanswerable questions still get a passage. This is how `RETRIEVAL_MAX_DISTANCE` is chosen.

This part costs only embeddings, a fraction of a cent, and repeated runs reuse cached embeddings.

### Answers

Each answering model answers every question through the real question service. Then:

| Metric | Meaning | How it is checked |
|---|---|---|
| Score | Correct = 1, partially correct = 0.5 | Judge model, against the reference answer |
| Faithful | Every claim is supported by the retrieved passages | Judge model, against the passages only |
| Citations supported | A cited passage contains the evidence | Phrase matching, no model |
| Correct refusals | Unanswerable questions answered with "not covered" | The system's own `answerable` flag |
| False refusals | Answerable questions it refused | The system's own `answerable` flag |
| Latency, tokens, cost | p50 and p95 time, tokens per question, cost per 100 questions | Measured, with prices from `evals/pricing.py` |

Correctness and faithfulness are separate on purpose. A correct but unfaithful answer means the model used outside knowledge, which will fail on documents it has never seen. A faithful but incorrect answer usually means retrieval missed the right passage.

### The judge

Grading free-text answers needs judgement, so a second model grades them ("LLM as judge"). To keep it trustworthy:

- The judge (`gpt-6.1-sol`) is a different, larger model than the ones being graded, since models tend to favour their own style.
- It sees the reference answer written by a person, not just the question.
- Faithfulness is judged against the retrieved passages only, not the judge's own knowledge.
- Every verdict is saved with a one-sentence explanation, and the report lists every question that was not fully correct, so the grading can be checked by hand.
- Refusals and citation checks need no judge at all.

### Question-type checks and second opinions

Two more layers sit on top of the metrics above, explained with examples in [Evaluation frameworks](evaluation-frameworks.md):

- **Question-type checks**, run on every evaluation with no model: exact figures for number and table questions, every part covered for multi-part questions, both companies named and cited for comparisons, and no figure borrowed from the other company for trap questions. With the shared metrics above, this makes the evaluation two-level: categories, and tests that belong to each category.
- **Second opinions from RAGAS and DeepEval** (optional): the same answers graded by two open-source frameworks with the same judge model, and a report of where all three graders agree and disagree.

## Running it

| Command | Does | Needs | Cost |
|---|---|---|---|
| `make eval-offline` | Whole pipeline with stand-in models | Postgres | Free |
| `make eval-retrieval` | Chunk-size, top-k and cut-off comparison | Postgres, `OPENAI_API_KEY` | Under $0.01 |
| `make eval` | Retrieval plus answers from `gpt-6-luna` and `gpt-6.1-terra`, graded by `gpt-6.1-sol` | Postgres, `OPENAI_API_KEY` | About $1 |
| `make eval-frameworks` | The above on 20 questions, plus RAGAS and DeepEval second opinions | Also `pip install -e ".[frameworks]"` | More than `make eval`: each framework makes several judge calls per answer |

Postgres can be the one from Docker Compose (`docker compose up -d db`). The suite uses its own database, `rag_eval`, created automatically, so it never touches application data. Results are written to `evals/results/<run id>/` as `report.md` and `results.json`.

Useful options (`python -m evals --help` lists all):

- `--chat-models gpt-6-luna,gpt-6.1-sol` to compare other models
- `--reasoning-effort medium` to see whether more thinking helps
- `--judge-model gpt-6.1-terra` for a cheaper judge
- `--limit 10` for a quick partial run

The offline run's numbers say nothing about quality: the hashing embedder only matches shared words and the stand-in model quotes a passage. It exists to check the pipeline, and runs in the test suite.

## Why these numbers

### About 60 questions

Enough that one question is under 2% of the score, so a real difference between two settings shows up above the noise, while a full run stays around $1 and a few minutes. Small differences (two or three points) between configurations are still within noise at this size and should not decide anything on their own.

### Chunk sizes 200, 400 and 800

The default (400) and one step either side, doubling each time, covering the 200 to 800 range where useful sizes usually fall. Overlap stays at 15% of the chunk size, the same share as the default 60 of 400.

### k of 1, 3, 5 and 8

Hit@1 shows whether the best chunk is right; 3, 5 and 8 show how much is gained by sending more context, which is how the default of 5 is checked against cheaper (3) and broader (8) choices.

## Limits

- The documents are policy pages in English. Results may differ for long technical PDFs, scanned documents or other languages, so a deployment for those should add matching documents and questions.
- The questions were written by one person; a second reviewer would catch ambiguous wording and reference answers.
- Prices in `evals/pricing.py` were correct when checked and need updating when they change.
