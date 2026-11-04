# Extraction

The first step turns an uploaded file into plain text. Code: `app/ingestion/extract.py`.

## How each format is read

| Format | How it is read | Page numbers |
|---|---|---|
| PDF | `pypdf`, one page at a time | Yes |
| DOCX | `python-docx`; paragraphs, then tables as `cell \| cell` rows | No |
| TXT, Markdown | Decoded as UTF-8, byte-order mark removed | No |

**PDFs keep page numbers** so every chunk knows its pages and answers can cite "page 14".

**Word files have no page numbers.** Pages only exist once a word processor lays the document out, so any number we computed could be wrong. No number is better than a wrong one.

## Clear errors instead of empty documents

A file that cannot produce text is marked `failed` with a message the user can act on:

| Situation | The message says |
|---|---|
| Password-protected PDF | Remove the password and upload again |
| Damaged PDF or DOCX | The file could not be read |
| More pages than `MAX_PAGES` | The page count and the limit |
| No text at all (usually a scanned PDF) | It needs OCR, which is not supported yet |

## Cleaning the text

PDF text is messy. `normalize_text()` tidies it without changing its meaning:

- `informa-` / `tion` across a line break becomes `information`. Real hyphens like `well-known` stay.
- Repeated spaces become one, and lines are trimmed.
- At most one blank line is kept between paragraphs, because the chunker uses blank lines to find paragraphs.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `MAX_PAGES` | 1000 | Largest PDF accepted |
| `MAX_UPLOAD_MB` | 20 | Largest upload |

Next: [Chunking](chunking.md)
