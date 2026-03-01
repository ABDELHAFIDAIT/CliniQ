from prometheus_client import Counter, Gauge, Histogram

# Metrics for monitoring API performance and usage

# Total number of HTTP requests, labeled by method, endpoint, and status code
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"]
)


# Duration of HTTP requests in seconds, labeled by method and endpoint
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "Duration of HTTP requests in seconds",
    ["method", "endpoint"]
)


# Total number of clinical queries processed
CLINICAL_QUERY_COUNT = Counter(
    "clinical_query_count",
    "Total number of clinical queries processed"
)


# Total latency of the RAG pipeline (Expansion + Retrieval + Rerank + LLM) in seconds
RAG_LATENCY = Histogram(
    "rag_process_duration_seconds",
    "Temps de traitement total du pipeline RAG (Expansion + Retrieval + Rerank + LLM)",
    buckets=(1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, float("inf")) # des buckets adaptés pour les temps de traitement RAG, avec un bucket infini pour les requêtes très longues
)


# Total number of errors, labeled by endpoint
ERROR_COUNT = Counter(
    "http_errors_total",
    "Total number of errors",
    ["endpoint"]
)