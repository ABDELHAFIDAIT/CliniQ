import os
import uuid
import asyncio
import logging
import shutil
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from app.api.deps import get_current_user
from app.services.ingest_service import ingest_service
from app.models.user import User




logger = logging.getLogger(__name__)


router = APIRouter()


@router.post("/ingest", status_code=status.HTTP_201_CREATED)
async def upload_and_ingest_document(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    if not file.filename.lower().endswith(".pdf") or file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Seuls les documents PDF officiels sont acceptés pour l'analyse clinique."
        )

    os.makedirs("data", exist_ok=True)
    unique_id = uuid.uuid4()
    temp_file_path = f"data/{unique_id}_{file.filename}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = await asyncio.to_thread(
            ingest_service.ingest, 
            temp_file_path, 
            file.filename
        )

        if result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("message")
            )

        return result

    except Exception as e:
        logger.error(f"Échec critique de l'ingestion pour {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur interne est survenue lors du traitement clinique du document."
        )

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)







@router.get("/chunks")
async def get_ingested_chunks(limit: int = 10, offset: Optional[str] = None, current_user: User = Depends(get_current_user)):
    try:
        result, next_offset = ingest_service.client.scroll(
            collection_name=ingest_service.collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False 
        )

        formatted_chunks = []
        for point in result:
            payload = point.payload or {}
            
            formatted_chunks.append({
                "id": point.id,
                "content": payload.get("content"),
                "metadata": {
                    "document": payload.get("document"),
                    "page_start": payload.get("page_start"),
                    "page_end": payload.get("page_end"),
                    "chapter": payload.get("chapter"),
                    "section": payload.get("section"),
                    "subsection": payload.get("subsection"),
                    "content_type": payload.get("content_type"),
                    "patient_population": payload.get("patient_population"),
                    "clinical_tags": payload.get("clinical_tags"),
                    "version": payload.get("version"),
                    "validated_by": payload.get("validated_by"),
                    "date": payload.get("date")
                }
            })

        return {
            "status": "success",
            "total_requested": limit,
            "chunks_found": len(formatted_chunks),
            "next_offset": next_offset,
            "data": formatted_chunks
        }

    except Exception as e:
        logger.error(f"Erreur lors du monitoring des métadonnées Qdrant : {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la récupération des métadonnées des segments."
        )