import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.db.base import Base
from app.db.session import get_db
from app.models.user import User
from app.models.query import Query
from app.core.security import get_password_hash
from app.services.rag_service import get_rag_service
from app.services.ingest_service import get_ingest_service

SQLALCHEMY_TEST_URL = "sqlite:///./test.db"

engine = create_engine(SQLALCHEMY_TEST_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Fixtures Base de données et utilisateurs de test


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db_session():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture()
def test_user(db_session):
    user = User(
        username="hafid01",
        email="hafid01@cliniq.com",
        hashed_password=get_password_hash("hafid_cliniq_01"),
        role="medecin",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    yield user

    try:
        db_session.query(Query).filter_by(user_id=user.id).delete(
            synchronize_session=False
        )
        db_session.delete(user)
        db_session.commit()
    except Exception:
        db_session.rollback()


@pytest.fixture()
def auth_headers(test_user):
    from app.core.security import create_access_token

    token = create_access_token(subject=test_user.username)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def mock_rag_service(sample_rag_result):
    mock = MagicMock()
    mock.ask = AsyncMock(return_value=sample_rag_result)
    return mock


@pytest.fixture()
def mock_ingest_service():
    mock = MagicMock()

    mock.ingest.return_value = {
        "status": "success",
        "document": "guide.pdf",
        "chunks_created": 42,
        "breakdown": {"text": 38, "table": 4},
    }

    point = MagicMock()
    point.id = "point_123"
    point.payload = {"content": "PARACETAMOL 500mg sur 8h", "document": "guide.pdf"}

    mock.client.scroll.return_value = ([point], None)
    mock.collection_name = "cliniq_docs"

    return mock


@pytest.fixture()
def client(db_session, mock_rag_service, mock_ingest_service):

    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    app.dependency_overrides[get_ingest_service] = lambda: mock_ingest_service

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# Fixtures pour les tests de RAG et d'évaluation


@pytest.fixture()
def sample_chunks():
    return [
        {
            "content": "En cas de diarrhée aiguë, administrer SRO 50ml/kg sur 4h.",
            "metadata": {
                "document": "guide_clinique.pdf",
                "page": 12,
                "chapter": "PÉDIATRIE",
                "section": "Diarrhée aiguë",
            },
            "score": 0.92,
            "rerank_score": 0.88,
        },
        {
            "content": "Signes de déshydratation sévère : pli cutané persistant, yeux enfoncés, soif intense.",
            "metadata": {
                "document": "guide_clinique.pdf",
                "page": 13,
                "chapter": "PÉDIATRIE",
                "section": "Déshydratation",
            },
            "score": 0.85,
            "rerank_score": 0.81,
        },
    ]


@pytest.fixture()
def sample_rag_result(sample_chunks):
    return {
        "answer": "En cas de diarrhée aiguë chez l'enfant, administrer SRO. (guide_clinique.pdf, p.12)",
        "metrics": {
            "answerrelevancy": 0.82,
            "faithfulness": 0.91,
            "contextualprecision": 0.75,
            "contextualrecall": 0.78,
        },
    }
