# Choosing the chunk size

## The trade-off

| Too small (e.g. 50 tokens) | Too large (e.g. 2,000 tokens) |
|---|---|
| Loses context ("that mirror", but which?) | Mixes topics, so it matches no single question well |
| The model gets scattered fragments | Irrelevant text fills the prompt, raising cost |
| Many more chunks to store and search | Fewer, blunter search results |

## Why 400 tokens to start

Useful sizes are usually between about 200 and 800 tokens. 400 sits in the middle and is well under the embedding model's input limit. The 60-token overlap is about one or two sentences.

## Measured, not guessed

400 is an educated default, not a final answer:

1. Start with the default.
2. Run the evaluation suite with several sizes, for example 200, 400 and 800.
3. Compare retrieval hit rate, answer quality, cost and speed, and keep the best.
4. Run it again whenever something changes: the models, the prompt, or the kind of documents users upload.

Because sizes are settings, trying a new one is a one-line change.

## Short and long documents

Chunk size does not depend on document length. A chunk holds one idea, and an idea is about the same size in a short note or a long manual. Only the number of chunks changes:

| Document | About | Chunks at 400 / 60 |
|---|---|---|
| Two paragraphs | 200 tokens | 1 |
| 10-page report | 5,000 tokens | 15 |
| 1,000-page manual | 500,000 tokens | 1,500 |

Long documents are handled by other means: a fast search index ([Embeddings](embeddings.md)), background processing ([Background processing](background-processing.md)), and the page and size limits.

The *kind* of document matters more than its length. Short FAQ entries may suit smaller chunks; long contracts may suit larger ones. Fancier strategies, such as section summaries, are left out until evaluation shows they are needed.

Next: [Embeddings](embeddings.md)
