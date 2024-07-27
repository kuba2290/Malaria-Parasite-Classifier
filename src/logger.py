import logging
import os
from datetime import datetime

# Define the log file path
LOG_FILE_PATH = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_FILE_PATH, exist_ok=True)

# Create a unique log file name with timestamp
LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE = os.path.join(LOG_FILE_PATH, LOG_FILE_NAME)

# Configure the logging
logging.basicConfig(
    filename=LOG_FILE,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Add this line to also print log messages to the console
logging.getLogger().addHandler(logging.StreamHandler())

# Example usage
if __name__ == "__main__":
    logging.info("Logging has been initialized")
    print(f"Log file created at: {LOG_FILE}")