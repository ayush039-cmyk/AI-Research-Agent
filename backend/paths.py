import os

from dotenv import load_dotenv

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')
FILES_DIR = os.path.join(PROJECT_ROOT, 'files')
DATA_DIR = os.path.join(BACKEND_DIR, 'data')
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)

load_dotenv(ENV_PATH, override=True)
