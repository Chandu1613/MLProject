import logging
import os

log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_message(message):
    logging.info(message)
