import logging
import os


LOGS_DIR = "logs"

os.makedirs(LOGS_DIR, exist_ok=True)


LOG_FILE = os.path.join(LOGS_DIR, "clinirag.log")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)


logger = logging.getLogger("CliniRAG")