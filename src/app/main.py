from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .api import router as api_router
from .api.response.response import ok, error, unexpect_error
from .logger import log
from .dependencies import get_cors_origins, get_milvus_wrapper, get_reranker
from .middleware import prove_server_identity, PUBLIC_KEY_HEADER, SIGNATURE_HEADER, SERVER_RANDOM_HEADER, TS_HEADER
from .config import get_settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Clear LRU cache for settings at startup
    get_settings.cache_clear()
    log.info("Cleared LRU cache for get_settings() at startup.")

    # Initialize MilvusWrapper to pre-load embedding model
    log.info("Attempting to pre-load embedding model by initializing MilvusWrapper...")
    get_milvus_wrapper()
    log.info("MilvusWrapper initialized and embedding model should be pre-loaded.")

    # Initialize Reranker
    log.info("Attempting to pre-load reranker model by initializing Reranker...")
    get_reranker()
    log.info("Reranker initialized and model should be pre-loaded.")

    yield

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[PUBLIC_KEY_HEADER, SIGNATURE_HEADER, SERVER_RANDOM_HEADER, TS_HEADER],
)

app.middleware("http")(prove_server_identity)

app.include_router(api_router)


@app.get("/")
async def root():
    return ok()


# Custom global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handle all uncaught exceptions globally.
    """
    # handle HTTPException
    if isinstance(exc, HTTPException):
        log.error(f"HTTPException: {exc.detail}")
        # return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return error(exc.detail)

    log.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content=unexpect_error())

