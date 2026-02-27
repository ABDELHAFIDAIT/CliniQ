import logging
from app.models.user import User
from app.models.query import Query
from sqlalchemy.orm import Session
from app.schemas.chat import ChatRequest
from app.db.session import get_db
from app.api.deps import get_current_user
from app.services.rag_service import rag_service
from fastapi import APIRouter, Depends, HTTPException, status


logger = logging.getLogger(__name__)


router = APIRouter()


@router.post("/ask", status_code=status.HTTP_200_OK)
async def ask_clinical_question(request: ChatRequest,  db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="La question ne peut pas être vide !"
        )

    try:
        result = await rag_service.ask(request.question)
        
        new_query = Query(
            query=request.question,
            response=result.get("answer"),
            user_id=current_user.id
        )
        
        db.add(new_query)
        db.commit()
        db.refresh(new_query)

        return result

    except Exception as e:
        logger.error(f"Erreur Chat API pour l'utilisateur {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur est survenue lors de la génération de la réponse médicale."
        )