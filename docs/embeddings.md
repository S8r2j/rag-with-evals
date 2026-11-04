# Embeddings

Code: `app/embeddings/`.

## What they are for

Search has to work by **meaning**, not exact words: "When was it built?" should find "founded in 1998". An embedding model turns each chunk into 1,536 numbers (a vector), and texts with similar meaning get similar vectors. A question is embedded the same way, and the nearest chunk vectors are the best matches.

## How they are stored and searched

- **Model:** OpenAI `text-embedding-3-small` by default. The reasons for this choice, and when another model fits better, will be documented with evaluation results.
- **Batches:** chunks are embedded in batches rather than one request each, which is much faster.
- **Next to the text:** each vector is saved in its chunk row (`vector(1536)` with pgvector), so one query returns text, pages and vector together.
- **Fast search:** an HNSW index finds the nearest vectors without comparing against every chunk, so search stays fast with hundreds of thousands of chunks. Distance is cosine, which compares meaning (direction) rather than length.
- **Model recorded:** each document stores which model made its vectors. Vectors from different models cannot be compared, so changing models means processing documents again.

## Why 1,536 numbers

1,536 is not a number we picked: it is the full output size of `text-embedding-3-small`, fixed when the model was trained. (The larger `text-embedding-3-large` produces 3,072.)

Each number captures some aspect of meaning that the model learned on its own. More numbers can tell apart texts that are close in meaning; fewer numbers are smaller and faster.

| More (e.g. 3,072) | Fewer (e.g. 512) |
|---|---|
| Finer differences in meaning | Slightly less accurate |
| About 12 KB per chunk | About 2 KB per chunk |
| Slower search, larger index | Faster search, smaller index |
| Above 2,000, pgvector's HNSW index needs the half-precision `halfvec` type | Fits the index easily |

These models are trained so the first numbers carry the most information, which means a shorter vector can be requested (for example 512) with only a small loss in quality.

**Why 1,536 here:** it keeps the model's full quality, fits pgvector's index without workarounds, and costs little at this scale (a 1,000-page manual is about 1,500 chunks, roughly 9 MB of vectors).

**How it gets settled:** like the chunk size, by evaluation. Run the test questions with 512, 1,024 and 1,536 and compare retrieval accuracy. If 512 scores almost the same, it is the better choice at large scale, at a third of the storage.

It is a constant in code (`EMBEDDING_DIMENSIONS`), not a setting, on purpose: changing it means changing the database column and re-embedding every chunk, so it should be a deliberate decision backed by evaluation results.

## Running without an API key

`EMBEDDING_PROVIDER=hashing` builds vectors from the words themselves. Tests and offline development use it. It does not understand meaning ("car" and "automobile" look unrelated), so it is never used to judge quality.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_PROVIDER` | `openai` | `openai`, or `hashing` for tests and offline work |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model name |
| `EMBEDDING_BATCH_SIZE` | 128 | Chunks per request; settings reject a batch that could exceed the provider's token cap |
| `OPENAI_API_KEY` | none | Needed for the `openai` provider |
