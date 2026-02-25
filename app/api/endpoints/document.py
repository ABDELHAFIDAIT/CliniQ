import os
import uuid
import asyncio
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from app.api.deps import get_current_user
from app.services.ingest_service import ingest_service
from app.models.user import User
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/ingest", status_code=status.HTTP_201_CREATED)
async def upload_and_ingest_pdf(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    
    if not file.filename.lower().endswith(".pdf") or file.content_type != "application/pdf" :
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Seuls les fichiers PDF sont acceptés pour l'ingestion."
        )

    os.makedirs("temp", exist_ok=True)
    temp_file_path = f"temp/{uuid.uuid4()}_{file.filename}"

    try:
        content = await file.read()
        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)

        num_chunks = await asyncio.get_event_loop().run_in_executor(
            None, ingest_service.ingest_pdf, temp_file_path, file.filename
        )

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_ingested": num_chunks,
            "message": f"Le document '{file.filename}' a été découpé en {num_chunks} segments sémantiques."
        }

    except Exception as e:
        logger.error(f"Erreur ingestion '{file.filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'ingestion !"
        )

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)