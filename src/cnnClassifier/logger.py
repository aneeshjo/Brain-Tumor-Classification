import logging
import os
from datetime import datetime
from pathlib import Path


# ==========================================================
# Create Log File Name
# ==========================================================

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# ==========================================================
# Create Logs Directory
# ==========================================================

# Creates a path object.
logs_path=Path("logs")

logs_path.mkdir(exist_ok=True)

LOG_FILE_PATH=logs_path / LOG_FILE

# ==========================================================
# Configure Logging
# ==========================================================

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)