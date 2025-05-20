import os
from datetime import datetime

LOG_PATH = "logs"
LOG_DATE = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = os.path.join(LOG_PATH, f"{LOG_DATE}_chat_history.log")

# Make sure the logs folder exists
os.makedirs(LOG_PATH, exist_ok=True)

def log_user_interaction(user_input, intent, confidence, bot_response):
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(f"{datetime.now()} | User: {user_input}\n")
        log_file.write(f"{datetime.now()} | Predicted Intent: {intent} (confidence: {confidence})\n")
        log_file.write(f"{datetime.now()} | Bot Response: {bot_response}\n")
        log_file.write("-" * 80 + "\n")
