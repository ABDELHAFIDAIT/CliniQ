import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_module_patches = [
    patch("sentence_transformers.SentenceTransformer", MagicMock()),
    patch("qdrant_client.QdrantClient", MagicMock()),
    patch("flashrank.Ranker", MagicMock()),
    patch("mlflow.set_tracking_uri", MagicMock()),
    patch("mlflow.set_experiment", MagicMock()),
    patch("mlflow.start_run", MagicMock()),
]
for _p in _module_patches:
    _p.start()

from app.main import app
from app.db.base import Base
from app.db.session import get_db
from app.models.user import User
from app.models.query import Query
from app.core.security import get_password_hash


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
def client(db_session):
    with patch(
        "app.services.rag_service.RAGService.__init__", return_value=None
    ), patch("app.services.ingest_service.IngestService.__init__", return_value=None):

        app.dependency_overrides[get_db] = lambda: db_session

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
