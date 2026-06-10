import os
from dotenv import load_dotenv

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')

# Vercel has read-only filesystem, only /tmp is writable
IS_VERCEL = os.getenv('VERCEL') == '1'
DATA_DIR = '/tmp/data' if IS_VERCEL else os.path.join(BACKEND_DIR, 'data')
FILES_DIR = '/tmp/files' if IS_VERCEL else os.path.join(PROJECT_ROOT, 'files')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)

ENV_PATH = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(ENV_PATH, override=True)
