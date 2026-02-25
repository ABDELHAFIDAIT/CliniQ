from fastapi import FastAPI
from app.core.config import settings
from app.api.router import router
from app.models.user import User
from app.models.query import Query


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for CliniQ application",
    version=settings.VERSION
)


app.include_router(router)


@app.get("/")
async def root() :
    return {
        "status": "success",
        "message": "Welcome to CliniQ API",
        "version": settings.VERSION
    }