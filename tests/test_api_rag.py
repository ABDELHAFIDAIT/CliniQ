import io
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.security import get_password_hash
from app.models.user import User
from app.models.query import Query



# Test d'inscription réussie
def test_signup_success(client) :
    response = client.post(
        "api/auth/signup",
        json={
            "username" : "hafid02",
            "email" : "hafid02@cliniq.com",
            "password" : "hafid_cliniq_02",
            "role" : "medecin"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "hafid02"
    assert data["email"] == "hafid02@cliniq.com"
    assert "hashed_password" not in data



# Test de connexion réussie
def test_login_success(client, test_user) :
    response = client.post(
        "api/auth/login",
        data={
            "username" : test_user.username,
            "password" : "hafid_cliniq_01"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"



# Fixture pour mocker le service RAG et éviter les appels externes lors des tests de l'endpoint /api/chat/ask
@pytest.fixture(autouse=True)
def mock_rag(sample_rag_result):
    with patch("app.api.endpoints.chat.rag_service") as mock:
        mock.ask = AsyncMock(return_value=sample_rag_result)
        yield mock



# Test de l'endpoint /api/chat/ask
def test_ask_success(client, auth_headers) :
    response = client.post(
        "/api/chat/ask",
        json={
            "question": "Comment traiter une diarrhée aiguë chez l'enfant ?"
        },
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "metrics" in data




# Test que les questions posées sont bien enregistrées dans la base de données
def test_ask_saves_query_in_db(client, auth_headers, db_session, test_user) :
    before = db_session.query(Query).filter_by(user_id=test_user.id).count()
    
    client.post(
        "api/chat/ask",
        json={
            "question": "Quels sont les signes de déshydratation ?"
        },
        headers=auth_headers
    )
    
    after = db_session.query(Query).filter_by(user_id=test_user.id).count()
    assert after == before + 1



# Fixture pour mocker le service d'ingestion et éviter les appels externes lors des tests de l'endpoint /api/document/ingest
@pytest.fixture() # autouse=True
def mock_ingest():
    with patch("app.api.endpoints.document.ingest_service") as mock:
        mock.ingest.return_value = {
            "status": "success",
            "document": "guide.pdf",
            "chunks_created": 42,
            "breakdown": {"text": 38, "table": 4}
        }
        yield mock


# Helper pour créer un fichier PDF factice à envoyer lors du test de l'endpoint d'ingestion
def pdf_file(filename="guide.pdf"):
        return {"file": (filename, io.BytesIO(b"%PDF-1.4 fake guide des protocles"), "application/pdf")}


# Test de l'endpoint /api/document/ingest
def test_ingest_success(client, auth_headers, mock_ingest) :
    response = client.post(
        "/api/documents/ingest",
        files=pdf_file(),
        headers=auth_headers
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    assert data["document"] == "guide.pdf"
    assert data["chunks_created"] == 42
    
    





@pytest.fixture() # autouse=True
def mock_qdrant_scroll():
    point = MagicMock()
    point.id = "point_123"
    point.payload = {
        "content": "PARACETAMOL 500mg sur 8h",
        "document": "guide.pdf",
        "page_start": 12,
        "page_end": 12,
        "chapter": "PÉDIATRIE",
        "section": "Diarrhée",
        "subsection": "",
        "content_type": "text",
        "patient_population": "pediatrie",
        "clinical_tags": ["diarrhée", "paracetamol"],
        "version": "1",
        "validated_by": "Hafid",
        "date": "2026"
    }
    with patch("app.api.endpoints.document.ingest_service") as mock:
        mock.client.scroll.return_value = ([point], None)
        mock.collection_name = "cliniq_docs"
        yield mock




def test_get_chunks_success(client, auth_headers, mock_qdrant_scroll    ) :
    response = client.get(
        "/api/documents/chunks",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_found"] == 1
    assert data["data"][0]["content"] == "PARACETAMOL 500mg sur 8h"