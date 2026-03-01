from prometheus_client import Counter, Gauge, Histogram

# Infrastructure HTTP ===========================================

HTTP_REQUESTS_TOTAL = Counter(
    "cliniq_http_requests_total",
    "Nombre total de requêtes HTTP",
    ["method", "endpoint", "status"]
)

HTTP_REQUEST_DURATION = Histogram(
    "cliniq_http_request_duration_seconds",
    "Durée des requêtes HTTP",
    ["method", "endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

ERROR_COUNT = Counter(
    "cliniq_http_errors_total",
    "Nombre total d'erreurs HTTP (4xx/5xx)",
    ["endpoint"]
)



# RAG applicatif ===========================================

CLINICAL_QUERY_COUNT = Counter(
    "cliniq_rag_queries_total",
    "Nombre total de questions posées au RAG"
)

RAG_LATENCY = Histogram(
    "cliniq_rag_latency_seconds",
    "Latence complète du pipeline RAG (expand + search + rerank + generate + eval)",
    buckets=[1.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0]
)

RAG_RETRIEVED_DOCS = Histogram(
    "cliniq_rag_retrieved_docs_count",
    "Nombre de documents récupérés après reranking",
    buckets=[1, 2, 3, 5, 8, 10]
)

RAG_NO_CONTEXT = Counter(
    "cliniq_rag_no_context_total",
    "Nombre de fois où aucun contexte n'a été trouvé"
)


# Qualité DeepEval ===========================================

EVAL_SCORE = Histogram(
    "cliniq_eval_score",
    "Score DeepEval par métrique",
    ["metric"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

EVAL_FAILURES = Counter(
    "cliniq_eval_failures_total",
    "Nombre de métriques DeepEval sous le seuil (< 0.7)",
    ["metric"]
)