import logging
import re
from typing import List, Dict, Any
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        logger.info(f"Initialisation RAG avec {settings.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.reranker = CrossEncoder(settings.RERANKER_MODEL)
        self.qdrant = QdrantClient(url=settings.QDRANT_URL)
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.llm = genai.GenerativeModel(model_name=settings.LLM_MODEL)
        
        self.collection = settings.QDRANT_COLLECTION_NAME
    
    
    def expand_query(self, query: str) -> str:
        prompt = f"""
            En tant qu'expert médical, génère 3 variantes de recherche (mots-clés ou phrases courtes) en français pour la question suivante : '{query}'.
            L'objectif est de trouver des protocoles dans un guide clinique. 
            Réponds uniquement avec les variantes, une par ligne.
        """
        try :
            response = self.llm.generate_content(prompt)
            variantes = response.text.strip().split('\n')
            variantes = [re.sub(r'^\d+[\.\-\s]+', '', v).strip() for v in variantes]
            return [query] + variantes
        except Exception as e:
            logger.error(f"Error during query expansion: {e}")
            return [query]
    
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        potential_tags = [kw for kw in ["diarrhée", "toux", "fièvre", "asthme"] if kw in query.lower()]
        query_vector = self.embedding_model.encode(query).tolist()
        search_filter = None
        
        if potential_tags:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.clinical_tags",
                        match=MatchAny(any=potential_tags)
                    )
                ]
            )

        results = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True
        ).points
        
        return [
            {
                "content": r.payload.get("content"), 
                "metadata": r.payload.get("metadata"), 
                "score": r.score
            } 
            for r in results
        ]
    

    async def ask(self, question: str) -> Dict[str, Any] :
        queries = self.expand_query(question)
        
        all_chunks = []
        for q in queries:
            all_chunks.extend(self.hybrid_search(q))
        
        unique_chunks_dict = {chunk['content']: chunk for chunk in all_chunks}
        chunks_list = list(unique_chunks_dict.values())
        
        if chunks_list:
            pairs = [[question, c["content"]] for c in chunks_list if c.get("content")]
            if not pairs:
                sorted_chunks = []
            else:
                rerank_scores = self.reranker.predict(pairs)
                for i, chunk in enumerate(chunks_list):
                    chunk["rerank_score"] = float(rerank_scores[i])
                
                sorted_chunks = sorted(chunks_list, key=lambda x: x["rerank_score"], reverse=True)[:settings.RETRIEVAL_TOP_K]
        else:
            sorted_chunks = []
        
        context_parts = []
        for c in sorted_chunks:
            meta = c.get("metadata") or {}
            doc = meta.get("document", "Document inconnu")
            page = meta.get("page", "?")
            chap = meta.get("chapter", "N/A")
            sec = meta.get("section", "N/A")
            content = c.get("content", "")
            
            context_parts.append(
                f"--- SOURCE: {doc} (Page {page}) ---\n"
                f"Chapitre: {chap} | Section: {sec}\n"
                f"{content}"
            )
            
        context = "\n\n".join(context_parts)
        
        if not context:
            return {
                "answer": "Désolé, je n'ai trouvé aucun protocole médical correspondant à votre recherche dans le guide.",
                "sources": []
            }
        
        system_prompt = f"""
            Tu es CliniQ, un assistant médical expert pour la Polynésie Française.
            Réponds à la question en te basant EXCLUSIVEMENT sur les protocoles ci-dessous.
            
            RÈGLES :
            1. Cite TOUJOURS le document et la page.
            2. Si la réponse n'est pas dans le contexte, dis que tu ne sais pas.
            3. Sois structuré (signes de gravité, traitement, conduite à tenir).
            
            CONTEXTE :
            {context}
            
            QUESTION :
            {question}
        """
        
        response = self.llm.generate_content(system_prompt)
        
        return {
            "answer": response.text,
            "sources": sorted_chunks
        }



rag_service = RAGService()