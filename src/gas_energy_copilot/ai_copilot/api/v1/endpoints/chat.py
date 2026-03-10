from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import structlog.stdlib

from pipeline_safety_rag_crew.crew import PipelineSafetyRAGCrew

router = APIRouter()
log = structlog.stdlib.get_logger()


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat", summary="Ask a pipeline safety question")
def chat(request: ChatRequest) -> ChatResponse:
    """Run the two-agent CrewAI crew and return a cited regulatory answer."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    log.info("chat_request", question=request.question[:120])
    try:
        result = PipelineSafetyRAGCrew().crew().kickoff(inputs={"question": request.question})
        log.info("chat_response_ok")
        return ChatResponse(answer=result.raw)
    except Exception as exc:
        log.error("chat_error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Crew execution failed: {exc}") from exc
