from fastapi import APIRouter

from app.api.deps import ChatModelDep, CurrentUser, DbSession, EmbedderDep, SettingsDep
from app.schemas.questions import AnswerRead, QuestionRequest, SourceRead, Usage
from app.services.questions import QuestionService

router = APIRouter(prefix="/questions", tags=["questions"])


@router.post("", response_model=AnswerRead)
async def ask_question(
    body: QuestionRequest,
    user: CurrentUser,
    db: DbSession,
    settings: SettingsDep,
    embedder: EmbedderDep,
    chat_model: ChatModelDep,
) -> AnswerRead:
    """Answer a question from your documents.

    Each claim in `answer` cites a source as [number]; `sources` lists every
    passage that was retrieved, with `cited` marking the ones the answer used.
    """
    result = await QuestionService(db, settings, embedder, chat_model).ask(
        user.id, body.question, document_ids=body.document_ids, top_k=body.top_k
    )
    return AnswerRead(
        answerable=result.answerable,
        answer=result.answer,
        sources=[
            SourceRead(
                number=source.number,
                document_id=source.scored.chunk.document_id,
                filename=source.scored.filename,
                page_start=source.scored.chunk.page_start,
                page_end=source.scored.chunk.page_end,
                chunk_index=source.scored.chunk.chunk_index,
                text=source.scored.chunk.text,
                distance=round(source.scored.distance, 4),
                cited=source.cited,
            )
            for source in result.sources
        ],
        model=result.model,
        usage=Usage(input_tokens=result.input_tokens, output_tokens=result.output_tokens),
        latency_ms=result.latency_ms,
    )
