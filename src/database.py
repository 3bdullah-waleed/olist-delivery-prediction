from sqlalchemy import create_engine
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
from src.logger import get_logger

logger = get_logger(__name__)


def get_engine():
    """Create and return a SQLAlchemy engine connected to the Olist database."""
    try:
        connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(connection_string)
        logger.info("Database engine created successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        raise
    