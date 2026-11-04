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

## Running without an API key

`EMBEDDING_PROVIDER=hashing` builds vectors from the words themselves. Tests and offline development use it. It does not understand meaning ("car" and "automobile" look unrelated), so it is never used to judge quality.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_PROVIDER` | `openai` | `openai`, or `hashing` for tests and offline work |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model name |
| `EMBEDDING_BATCH_SIZE` | 128 | Chunks per request; settings reject a batch that could exceed the provider's token cap |
| `OPENAI_API_KEY` | none | Needed for the `openai` provider |
