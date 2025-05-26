from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .api import router as api_router
from .api.response.response import ok, error, unexpect_error
from .logger import log
from .dependencies import get_cors_origins
from .middleware import prove_server_identity
import nltk

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Download NLTK resources, used for keyword extraction at RAG
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)

    yield

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

