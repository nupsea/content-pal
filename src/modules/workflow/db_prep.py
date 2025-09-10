import os
from dotenv import load_dotenv

os.environ['RUN_TIMEZONE_CHECK'] = '0'

from src.modules.workflow.db import init_db

load_dotenv()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()

    print("Database initialized successfully.")