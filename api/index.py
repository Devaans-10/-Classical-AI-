"""Vercel Serverless Function entrypoint for FastAPI application.
"""

from pathlib import Path
import sys

# Ensure project root is in Python sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.backend.main import app
