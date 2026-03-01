import re
import time
import logging
import mlflow
import mlflow.langchain
from typing import List, Dict, Any
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from app.core.config import settings
from langchain_ollama import ChatOllama
from app.core.metrics import CLINICAL_QUERY_COUNT
from app.services.eval_service import eval_service
from app.core.metrics import RAG_LATENCY, RAG_RETRIEVED_DOCS, RAG_NO_CONTEXT, EVAL_SCORE, EVAL_FAILURES, CLINICAL_QUERY_COUNT



logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
            # mlflow.langchain.autolog()
        except Exception as e:
            logger.warning(f"MLflow unavailable, tracking disabled: {e}")
        
        logger.info(f"Initialisation RAG avec {settings.EMBEDDING_MODEL}")
        
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.reranker = CrossEncoder(settings.RERANKER_MODEL)
        self.qdrant = QdrantClient(url=settings.QDRANT_URL)
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.llm = genai.GenerativeModel(model_name=settings.LLM_MODEL)
        
        self.collection = settings.QDRANT_COLLECTION_NAME
    
        self.reference_llm = ChatOllama(
            model=settings.EVAL_LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0
        )
    
    
    @mlflow.trace
    async def generate_reference_answer(self, question: str, context: str) -> str:
        try:
            ref_prompt = f"""
                En tant qu'expert médical, rédige une réponse EXHAUSTIVE et EXACTE à la question suivante en utilisant le contexte fourni. 
                Ta réponse servira de référence de vérité terrain.
                
                CONTEXTE : {context}
                
                QUESTION : {question}
                
                RÉPONSE DE RÉFÉRENCE :
            """
            
            res = await self.reference_llm.ainvoke(ref_prompt)
            content = res.content.strip()
            return str(content)
        
        except Exception as e:
            logger.error(f"Erreur Ollama Reference : {e}")
            return ""
    
    
    
    def format_docs(self, chunks: List[Dict[str, Any]]) :
        formatted = []
        for c in chunks:
            m = c.get("metadata") or {}
            header = f"[DOC: {m.get('document')} | PAGE: {m.get('page')} | SECTION: {m.get('section')}]"
            content = c.get("content", "")
            formatted.append(f"{header}\n{content}")
        return "\n\n".join(formatted)
    
    
    @mlflow.trace
    async def evaluate_performance(self, question: str, response: str, retrieved_docs: List[Any]):
        try:
            context_text = self.format_docs(retrieved_docs)
            context_list = [c["content"] for c in retrieved_docs]

            expected_output = await self.generate_reference_answer(question, context_text)
            
            if not expected_output:
                logger.warning("Évaluation annulée : expected_output vide.")
                return
            
            
            test_case = LLMTestCase(
                input=question,
                actual_output=response,
                expected_output=expected_output,
                retrieval_context=context_list
            )
            
            metrics = [
                AnswerRelevancyMetric(threshold=0.7, model=eval_service),
                FaithfulnessMetric(threshold=0.7, model=eval_service),
                ContextualPrecisionMetric(threshold=0.7, model=eval_service),
                ContextualRecallMetric(threshold=0.7, model=eval_service)
            ]

            results = {}
            for metric in metrics:
                try:
                    await metric.a_measure(test_case)
                    name = metric.__class__.__name__.replace('Metric', '').lower()
                    results[name] = float(metric.score)
                    
                    EVAL_SCORE.labels(metric=name).observe(results[name])
                    if results[name] < 0.7:
                        EVAL_FAILURES.labels(metric=name).inc()
                except Exception as metric_err:
                    name = metric.__class__.__name__.replace('Metric', '').lower()
                    logger.warning(f"Métrique {name} échouée : {metric_err}")
                
            mlflow.log_metrics(results)
            
            mlflow.log_text(expected_output, "evaluation/ollama_reference.txt")
                
            mlflow.log_metrics(results)
            
            mlflow.log_text(expected_output, "evaluation/ollama_reference.txt")
            
            logger.info(f"Métriques DeepEval enregistrées avec succès !, {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation DeepEval : {e}")
    
    
    @mlflow.trace
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
    

    @mlflow.trace
    async def ask(self, question: str) -> Dict[str, Any] :
        CLINICAL_QUERY_COUNT.inc()
        rag_start = time.time()
        
        with mlflow.start_run(run_name="RAG_CLINIQ") :
            
            mlflow.set_tags({
                "project": "CliniQ",
                "architecture": "Hybrid RAG",
                "llm_provider": "GEMINI",
                "reranker": "CrossEncoder",
                "evaluation": "DeepEval"
            })
                
            mlflow.log_params({
                "llm_model": settings.LLM_MODEL,
                "temperature": 0,
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "embedding_model": settings.EMBEDDING_MODEL,
                "retrieval_k": 10,
                "similarity": "cosine",
                "reranker_model": settings.RERANKER_MODEL,
                "eval_model": settings.EVAL_LLM_MODEL,
                "eval_threshold": 0.7,
                "qdrant_collection": settings.QDRANT_COLLECTION_NAME
            })
            
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
            
            RAG_RETRIEVED_DOCS.observe(len(sorted_chunks))
            
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
                
            mlflow.log_text(str(sorted_chunks), "artifacts/retrieved_docs.txt")    
            mlflow.log_metric("retrieved_docs_count", len(sorted_chunks))    
            
            context = "\n\n".join(context_parts)
            
            if not context:
                RAG_NO_CONTEXT.inc()
                RAG_LATENCY.observe(time.time() - rag_start)
                
                return {
                    "answer": "Désolé, je n'ai trouvé aucun protocole médical correspondant à votre recherche dans le guide.",
                    "sources": [],
                    "evaluation": None
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
            
            mlflow.log_text(str(question), "inputs/question.txt")
            mlflow.log_text(str(response.text), "outputs/ai_response.txt")
            mlflow.log_text(str(context), "artifacts/context_used.txt")
            
            metrics = await self.evaluate_performance(question, response.text, sorted_chunks)
            
            RAG_LATENCY.observe(time.time() - rag_start)
            
            return {
                "answer": response.text,
                "metrics": metrics
            }



rag_service = RAGService()