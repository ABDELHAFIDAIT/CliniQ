import os
import re
import uuid
import logging
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings



logger = logging.getLogger(__name__)

class IngestService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        self.client = QdrantClient(
            url=settings.QDRANT_URL
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME

    
    
    def _prepare_collection(self, vector_size: int):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, 
                    distance=models.Distance.COSINE
                ),
            )



    def ingest_pdf(self, file_path: str, filename: str) :
        md_pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)
        
        headers_to_split_on = [
            ("#", "chapitre"),
            ("##", "section"),
            ("###", "sous-section"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        self._prepare_collection(vector_size=1024)
        
        total_points = 0
        batch_size = 200
        batch = []
        
        for page in md_pages:
            page_text = page["text"]
            page_num = page["metadata"]["page"]
            
            chunks = markdown_splitter.split_text(page_text)
            chunks = [c for c in chunks if c.page_content.strip()] 
            
            if not chunks:
                continue
            
            texts = [c.page_content for c in chunks]
            
            try:
                vectors = self.embeddings.embed_documents(texts)
            except Exception as e:
                logger.error(f"Embedding failed on page {page_num}: {e}")
                continue
                
            for chunk, vector in zip(chunks, vectors):
                content_type = "table" if re.search(r'\|.+\|', chunk.page_content) else "narrative text"
                
                metadata = {**chunk.metadata, "document": filename, "page": page_num + 1, "type": content_type}
                
                batch.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"content": chunk.page_content, "metadata": metadata}
                ))
                
                if len(batch) >= batch_size:
                    self.client.upsert(
                        collection_name=self.collection_name, points=batch
                    )
                    total_points += len(batch)
                    batch = []
            
        if batch:
            self.client.upsert(collection_name=self.collection_name, points=batch)
            total_points += len(batch)

        logger.info(f"Ingested {total_points} chunks from '{filename}'")
        
        return total_points