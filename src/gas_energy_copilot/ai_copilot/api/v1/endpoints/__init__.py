from fastapi import APIRouter

from gas_energy_copilot.ai_copilot.api.v1.endpoints.chat import router as chat_router

api_router = APIRouter()
api_router.include_router(chat_router, tags=["chat"])
