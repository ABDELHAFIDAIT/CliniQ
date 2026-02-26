from fastapi import APIRouter
from app.api.endpoints import auth, document

router = APIRouter()

router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(document.router, prefix="/documents", tags=["documents"])
