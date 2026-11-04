# Documentation

What happens to a file after it is uploaded:

```mermaid
flowchart LR
    U[Upload] --> Q[(Queue)]
    Q --> W[Worker]
    W --> X[Extract and clean]
    X --> C[Chunk]
    C --> E[Embed]
    E --> R[Ready]
```

Each page covers one step, in order:

| Page | What it explains |
|---|---|
| [Extraction](extraction.md) | Reading text out of PDF, Word and text files, and cleaning it |
| [Chunking](chunking.md) | Splitting text into small pieces, with a worked example |
| [Choosing the chunk size](chunk-size.md) | Why 400 tokens, how evaluation decides it, short vs long documents |
| [Embeddings](embeddings.md) | Turning chunks into vectors so search works by meaning |
| [Background processing](background-processing.md) | The queue and worker, retries, and what happens when things fail |
| [Live status updates](live-status.md) | Following a document's progress with server-sent events |
