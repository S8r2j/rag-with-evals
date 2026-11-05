# Evaluation documents: sources and licences

The documents in `evals/documents/` are pages from two openly licensed company handbooks, used as a fixed test set. They are produced by `python -m evals.prepare_documents`, which downloads each page at the commit below.

| Document | Source | Licence |
|---|---|---|
| `posthog-time-off.md`, `posthog-benefits.md`, `posthog-offboarding.md`, `posthog-side-gigs.md`, `posthog-training.md`, `posthog-grievances.md` | [PostHog/posthog.com](https://github.com/PostHog/posthog.com/tree/4fc3246084b49b31922fd4fc9482505b12367efe/contents/handbook/people) at `4fc3246` | MIT, see `posthog-handbook-MIT.txt` (Copyright (c) 2020-2025 PostHog Inc.) |
| `sourcegraph-wellness.md`, `sourcegraph-learning-and-development.md`, `sourcegraph-equity-refresh.md` | [sourcegraph/handbook](https://github.com/sourcegraph/handbook/tree/5752928b213a3ddbc7cee4459524f404a5b96fb4/content/departments/people-talent) at `5752928` | Apache 2.0, see `sourcegraph-handbook-Apache-2.0.txt` |

## Changes made

The wording is unchanged. Only website-specific markup was removed or converted:

- YAML front matter removed.
- MDX components replaced by their text: `<PrivateLink url="...">text</PrivateLink>` becomes `text`, and `<SmallTeam slug="people" />` becomes `People team`.
- An HTML table converted to a Markdown table, and remaining HTML tags removed.

The handbooks describe each company's policies at the time of the commit and may since have changed. They are used here only as realistic test material.
