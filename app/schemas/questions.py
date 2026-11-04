import uuid

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        max_length=50,
        description="Search only these documents. Omit to search all your documents.",
    )
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="How many passages to retrieve."
    )


class SourceRead(BaseModel):
    number: int = Field(description="The [number] used for this source in the answer.")
    document_id: uuid.UUID
    filename: str
    page_start: int | None
    page_end: int | None
    chunk_index: int
    text: str
    distance: float = Field(description="Cosine distance to the question; lower is closer.")
    cited: bool


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class AnswerRead(BaseModel):
    answerable: bool
    answer: str
    sources: list[SourceRead]
    model: str | None = Field(description="Chat model used; null when no model was called.")
    usage: Usage
    latency_ms: float
