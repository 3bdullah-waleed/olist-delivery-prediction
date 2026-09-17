import logging
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # also prints to console
    ]
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, consistent across all modules."""
    return logging.getLogger(name)


import src.features
print(src.features.__file__)