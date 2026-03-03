from pydantic import BaseModel
from datetime import datetime


class ChatRequest(BaseModel):
    question: str


class QueryHistory(BaseModel):
    id: int
    query: str
    response: str
    created_at: datetime

    class Config:
        from_attributes = True
