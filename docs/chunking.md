# Chunking

Code: `app/ingestion/chunking.py`.

## Why split documents at all

To answer a question, the relevant parts of the user's documents are sent to a language model. Sending whole documents does not work: they may not fit, they cost more, and answers get worse when the useful passage is buried in unrelated text.

So each document is split into small pieces called chunks. For each question, the few best-matching chunks are found and only those are sent.

## Sizes are counted in tokens

Models read and bill text in tokens (about three-quarters of a word). Chunks are measured with the same tokenizer family the models use (`o200k_base` via `tiktoken`), so a chunk's size in a prompt is known exactly.

## How text is packed

1. Break the text into the biggest natural pieces that fit in a chunk: paragraphs first, sentences if a paragraph is too long, and raw token windows only if a single sentence is too long.
2. Add pieces to the current chunk until the next one would not fit, then close it.
3. Start the next chunk with the last sentence or two of the previous one (the **overlap**).
4. Record the first and last page each chunk covers.

## Worked example

Limit: 30 words, overlap: about one sentence.

> ¹The Himalayan Observatory was founded in 1998. ²It sits at an altitude of 4,500 metres, which keeps it above most clouds. ³Its main telescope has a 2-metre mirror. ⁴That mirror was made in Germany and took three years to polish. ⁵Observations run from October to April. ⁶Monsoon rain makes the summer months too cloudy to use.

| Chunk | Sentences | Words |
|---|---|---|
| 0 | 1, 2, 3 | 28 |
| 1 | **3**, 4, 5 | 25 |
| 2 | **5**, 6 | 16 |

Adding sentence 4 to chunk 0 would make 40 words, so chunk 0 closes there. Bold sentences are repeated overlap.

## Why overlap matters

Sentence 4 alone says "That mirror was made in Germany", which does not say which mirror. Because of the overlap, chunk 1 also holds "Its main telescope has a 2-metre mirror", so the question *"Where was the telescope mirror made?"* finds the answer with its context.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `CHUNK_MAX_TOKENS` | 400 | Largest chunk |
| `CHUNK_OVERLAP_TOKENS` | 60 | Tokens repeated from the previous chunk; must be smaller than the maximum |
| `TOKENIZER_ENCODING` | `o200k_base` | Tokenizer used to count tokens |

Next: [Choosing the chunk size](chunk-size.md)
