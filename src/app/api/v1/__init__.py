from fastapi import APIRouter
from .openai import router as openai_router
from .summary import router as summary_router
from .info import router as info_router

router = APIRouter(prefix="/v1")
router.include_router(openai_router)
router.include_router(summary_router)
router.include_router(info_router)
