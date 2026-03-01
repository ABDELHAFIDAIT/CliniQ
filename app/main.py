import time
from fastapi import FastAPI, Response, Request
from app.core.config import settings
from app.api.router import router
from app.models.user import User
from app.models.query import Query
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from app.core.metrics import HTTP_REQUESTS_TOTAL, HTTP_REQUEST_DURATION, ERROR_COUNT, CLINICAL_QUERY_COUNT, RAG_LATENCY


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for CliniQ application",
    version=settings.VERSION
)




@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    path = request.url.path
    method = request.method
    
    response = await call_next(request)
    duration = time.time() - start_time

    HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=path, status=response.status_code).inc()
    HTTP_REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)
    
    if response.status_code >= 400:
        ERROR_COUNT.labels(endpoint=path).inc()

    return response




app.include_router(router, prefix=settings.API_STR)


@app.get("/")
async def root() :
    return {
        "status": "success",
        "message": "Welcome to CliniQ API",
        "version": settings.VERSION
    }
    
    
    
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)