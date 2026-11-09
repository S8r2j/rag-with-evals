# Evaluation frameworks

How this project grades answers in three independent ways, what each framework measures, and how to read the comparison between them. Code: `evals/checks.py` and `evals/frameworks/`. For the test set and the main metrics, see [Evaluation](evaluation.md).

## One example throughout

A user asks: *"What is PostHog's annual training budget, and can it be exceeded?"*

- The retrieved passage says: "Each person has a **$1000 per calendar year** training budget... it's not a hard limit, you can **request an increase to your budget in Brex**."
- The reference answer, written by a person: "$1000 per calendar year, but it is not a hard limit: people can request an increase in Brex."
- The system answers: "PostHog gives **$1000 per calendar year** for training [1]."

The answer is right about the amount but leaves out the second half of the question. Each grader below sees this answer differently.

The scores given for this example below are worked out by hand to show what each metric means; they are not output from a run.

## Two roles a model can play

| Role | Job | Here |
|---|---|---|
| The model being tested | Writes the answer | `gpt-6-luna`, `gpt-6.1-terra` |
| The judge | Grades the answer | `gpt-6.1-sol` for every grader, so differences come from how each framework measures, not from a different model |

Some checks need no model at all, and those are the most trustworthy because they cannot be wrong in a creative way.

## Level 1: checks every question gets

From [Evaluation](evaluation.md): did search find the evidence sentence (exact text match), is the answer correct and faithful (judge model), does a cited passage contain the evidence (exact text match), did it refuse when it should.

For the example: evidence found, citation supported, faithful. The judge would likely say **partially correct**, since the "can it be exceeded" part is missing.

## Level 2: checks for each type of question

Some mistakes only matter for one kind of question, and a lenient judge can let them through. These checks are plain code in `evals/checks.py`; the terms they look for are stored with each question (`must_include`, `must_not_include`).

| Question type | Check | Simple example of a failure |
|---|---|---|
| number, table | `exact_figures` | Asked for maternity leave after 12 months; answer says "up to 20 weeks" instead of 24 |
| multi_part, comparison | `covers_every_part` | The example above: "1,000" is there, "not a hard limit / increase" is not, so it fails with "missing not a hard limit" |
| comparison | `names_both_companies` | "How do PostHog and Sourcegraph support learning?" answered only about PostHog |
| comparison | `cites_both_documents` | Both companies named, but every citation points at PostHog pages |
| unanswerable (traps) | `no_borrowed_figure` | "How many days off does Sourcegraph give?" answered with PostHog's "25 days" |

Numbers are matched as whole numbers, so "5" does not match "15", citation markers like "[3]" are ignored, and "3,400" equals "3400". Alternatives are written with `|`, for example `4 weeks|four weeks`. A test checks that every reference answer passes its own checks, so a typo in the dataset is caught.

This is what makes the suite two-level: categories, and tests that belong to each category.

## Level 3: second opinions from other frameworks

Instead of trusting one judge, the same answers are graded by two widely used open-source frameworks, and the report shows where all three agree.

### RAGAS

A library of metrics built specifically for RAG. Its approach: break text into individual claims, then check each claim.

| Metric | Question it answers | For the example |
|---|---|---|
| `faithfulness` | What share of the answer's claims is supported by the passages? | The answer has one claim ("$1000 per calendar year"), and the passage supports it: **1.0** |
| `context_recall` | What share of the reference answer can be found in the passages? | Both "$1000 per year" and "can request an increase" are in the passage: **1.0**. Retrieval did its job. |
| `factual_correctness` (F1) | How well do the answer's claims match the reference's claims? | One of two reference claims covered, nothing wrong added: roughly **0.67** |

Read together: retrieval was fine and nothing was made up, but the answer was incomplete. That points at the answering model or the prompt, not at search.

### DeepEval

A framework that treats LLM checks like unit tests: every metric has a score and a pass threshold, and it can run under pytest in CI.

| Metric | Question it answers | For the example |
|---|---|---|
| `FaithfulnessMetric` | Same idea as RAGAS faithfulness, implemented independently | About **1.0** |
| `ContextualRecallMetric` | Same idea as RAGAS context recall | About **1.0** |
| `GEval` (correctness) | Does the answer meet criteria written in plain English? | Criteria: "penalise wrong figures, facts about the wrong company, and missing key facts". A missing key fact gives a middling score, around **0.5** |

G-Eval is the interesting one to learn: instead of writing a grader, you describe what good looks like in a sentence, and the judge model turns it into evaluation steps. The criteria used here are in `evals/frameworks/deepeval_grader.py`.

### Why use two frameworks that measure the same thing?

Because a grader is also software that can be wrong. If RAGAS and DeepEval both say an answer is unfaithful but the project's judge says it is faithful, one of them is mistaken, and that question is worth reading by hand. If all three agree on most questions, the grading can be trusted more than any single one of them.

## Reading the comparison in the report

Each grader's scores are turned into pass or fail (thresholds in `evals/frameworks/agreement.py`):

| Aspect | Own suite | RAGAS | DeepEval |
|---|---|---|---|
| faithfulness | judge says faithful | faithfulness at least 0.8 | faithfulness at least 0.8 |
| correctness | judge says correct or partially correct | factual F1 at least 0.5 | G-Eval at least 0.5 |
| retrieval | evidence sentence retrieved (exact match) | context recall at least 0.5 | contextual recall at least 0.5 |

The report then shows:

1. **Average scores** per framework and metric, with how many questions were scored.
2. **Agreement**: for every pair of graders and every aspect, the share of questions with the same verdict.
3. **Disagreements**: every question where the graders differ, to read by hand.

How to read agreement:

- **High agreement (around 90% or more)**: the graders see the same thing; the scores can be trusted.
- **One framework disagrees with the other two**: look at that framework's verdicts on the listed questions; it may handle a pattern badly, such as answers with citation markers.
- **Retrieval agreement is a special case**: the own suite's verdict here is an exact text match, not an opinion. Low agreement means the model-based recall metrics are unreliable on this data, which is worth knowing before relying on them elsewhere.

### Why the thresholds

- **0.8 for faithfulness**: a faithful answer should have every claim supported, so a score near 1.0 is expected. 0.8 allows for a grader splitting a sentence into claims awkwardly, without accepting an answer with real unsupported claims.
- **0.5 for correctness and recall**: these are partial-credit scales. Half or more corresponds to the own judge's "partially correct" or better, so the verdicts are comparable.

## Real-world problems found while integrating

Both libraries needed care, which is typical of fast-moving tooling and worth knowing before using them:

- **RAGAS 0.4.3 imports a module that `langchain-community` 0.4 removed.** The `frameworks` extra pins `langchain-community` below 0.4.
- **RAGAS does not recognise `gpt-6.1-sol` as a reasoning model.** It reads the version with `int()`, so "6.1" fails, and it sends `max_tokens` and `temperature=0.01`, which reasoning models reject. The adapter corrects the parameters (`use_reasoning_model_parameters`).
- **DeepEval does not know `gpt-6.1-sol` either**, and would send a temperature to it. The adapter plugs in a custom model class, DeepEval's extension point for any provider, which sends only the parameters we choose and returns structured output.
- **Both libraries send anonymous usage data by default.** The adapters opt out (`RAGAS_DO_NOT_TRACK`, `DEEPEVAL_TELEMETRY_OPT_OUT`).

Tests capture the exact requests each library would send, without calling any model, so a library update that breaks these fixes fails CI.

## Running it

```bash
pip install -e ".[frameworks]"     # RAGAS and DeepEval are optional
make eval-frameworks               # first 20 questions, all graders
python -m evals --frameworks ragas,deepeval            # every question
python -m evals --frameworks ragas --framework-model gpt-6.1-terra   # cheaper judge
```

The frameworks grade the answers of the first answering model (`--chat-models`). Question-type checks run on every evaluation and need no extra install.

**Cost.** RAGAS and DeepEval make several judge calls per answer, because they split answers into claims and check each one. Grading all 54 answerable questions with both frameworks costs noticeably more than the main evaluation; `make eval-frameworks` limits the run to 20 questions for learning and spot checks, and `--framework-model gpt-6.1-terra` halves the price per token.

## Other frameworks worth knowing

Not integrated here, but common in production:

| Tool | What it is for | Simple example |
|---|---|---|
| promptfoo | Comparing prompts or models side by side from a config file, often in CI | A table of 60 questions by 3 models, each cell pass or fail |
| LangSmith | Tracing every step of a live LLM app, and running evals on recorded traces | Click one user question to see the retrieved passages, prompt, answer, time and cost |
| Arize Phoenix | Open-source tracing and evals, self-hosted | Same idea as LangSmith, running on your own servers |
| OpenAI Evals | A framework and registry of evals for OpenAI models | Running a shared benchmark against a new model version |

The common thread: an evaluation is only as good as its grader, so use exact checks where possible, judge models where meaning matters, and compare graders before trusting any one of them.
