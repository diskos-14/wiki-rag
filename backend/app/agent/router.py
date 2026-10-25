from typing import Annotated

from fastapi import APIRouter, Depends

from app.agent.dependencies import get_agent_service
from app.agent.schemas import AgentAskResponse
from app.agent.service import AgentService
from app.core.security import require_api_key
from app.rag.schemas import AskRequest

router = APIRouter(tags=["agent"], dependencies=[Depends(require_api_key)])


@router.post("/agent/ask", response_model=AgentAskResponse, response_model_exclude_none=True)
async def agent_ask(
    request: AskRequest, service: Annotated[AgentService, Depends(get_agent_service)]
) -> AgentAskResponse:
    """Answer with the multi-step agent: understand → research (tools) → grounding check.

    Handles comparisons and questions spanning several articles. Slower than `/ask`
    (several LLM calls). Use `debug=true` to see each step. Shares conversations with `/ask`.
    """
    return await service.ask(
        request.question, conversation_id=request.conversation_id, debug=request.debug
    )
