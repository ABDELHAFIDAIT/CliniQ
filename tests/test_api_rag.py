import io
from app.models.query import Query


# Test d'inscription réussie
def test_signup_success(client):
    response = client.post(
        "api/auth/signup",
        json={
            "username": "hafid02",
            "email": "hafid02@cliniq.com",
            "password": "hafid_cliniq_02",
            "role": "medecin",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "hafid02"
    assert data["email"] == "hafid02@cliniq.com"
    assert "hashed_password" not in data


# Test de connexion réussie
def test_login_success(client, test_user):
    response = client.post(
        "api/auth/login",
        data={"username": test_user.username, "password": "hafid_cliniq_01"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


# Test de l'endpoint /api/chat/ask
def test_ask_success(client, auth_headers):
    response = client.post(
        "/api/chat/ask",
        json={"question": "Comment traiter une diarrhée aiguë chez l'enfant ?"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "metrics" in data


# Test que les questions posées sont bien enregistrées dans la base de données
def test_ask_saves_query_in_db(client, auth_headers, db_session, test_user):
    before = db_session.query(Query).filter_by(user_id=test_user.id).count()

    client.post(
        "api/chat/ask",
        json={"question": "Quels sont les signes de déshydratation ?"},
        headers=auth_headers,
    )

    after = db_session.query(Query).filter_by(user_id=test_user.id).count()
    assert after == before + 1


# Helper pour créer un fichier PDF factice à envoyer lors du test de l'endpoint d'ingestion
def pdf_file(filename="guide.pdf"):
    return {
        "file": (
            filename,
            io.BytesIO(b"%PDF-1.4 fake guide des protocles"),
            "application/pdf",
        )
    }


# Test de l'endpoint /api/document/ingest
def test_ingest_success(client, auth_headers):
    response = client.post(
        "/api/documents/ingest", files=pdf_file(), headers=auth_headers
    )

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    assert data["document"] == "guide.pdf"
    assert data["chunks_created"] == 42


def test_get_chunks_success(client, auth_headers):
    response = client.get("/api/documents/chunks", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_found"] == 1
    assert data["data"][0]["content"] == "PARACETAMOL 500mg sur 8h"
