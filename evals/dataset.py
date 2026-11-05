"""The evaluation questions and the documents they are about."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

EVALS_DIR = Path(__file__).parent
DOCUMENTS_DIR = EVALS_DIR / "documents"
DATASET_PATH = EVALS_DIR / "dataset" / "questions.jsonl"

Category = Literal["fact", "number", "table", "multi_part", "comparison", "unanswerable"]


class EvalQuestion(BaseModel):
    id: str
    category: Category
    question: str
    reference_answer: str
    # Documents that contain the answer, and short phrases copied word for
    # word from them that prove it. Retrieval is checked by looking for these
    # phrases in the retrieved chunks, which needs no model at all.
    documents: list[str]
    evidence: list[str]

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"

    @model_validator(mode="after")
    def _evidence_matches_category(self) -> "EvalQuestion":
        if self.answerable and not (self.documents and self.evidence):
            raise ValueError(f"{self.id}: answerable questions need documents and evidence")
        if not self.answerable and (self.documents or self.evidence):
            raise ValueError(f"{self.id}: unanswerable questions have no documents or evidence")
        return self


def load_questions(path: Path = DATASET_PATH) -> list[EvalQuestion]:
    questions = [
        EvalQuestion.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [q.id for q in questions]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate question ids: {sorted(duplicates)}")
    return questions


def document_paths() -> list[Path]:
    return sorted(DOCUMENTS_DIR.glob("*.md"))
