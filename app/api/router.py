from fastapi import APIRouter

from app.api.routes import auth, documents, questions

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router)
api_router.include_router(documents.router)
api_router.include_router(questions.router)
