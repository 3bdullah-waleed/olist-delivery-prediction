from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "prediction_requests_total", "Total number of prediction requests"
)
ERROR_COUNT = Counter(
    "prediction_errors_total", "Total number of failed prediction requests"
)
LATENCY_HISTOGRAM = Histogram(
    "prediction_latency_ms", 
    "Prediction request latency in milliseconds",
    buckets=[10, 25, 50, 100, 200, 500, 1000, 2000, float("inf")]
)


def get_metrics():
    """Return current metrics in Prometheus text format."""
    return generate_latest()