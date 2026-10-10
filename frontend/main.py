"""
Frontend Entry Point
=====================
Thin wrapper that adds the repo root to sys.path so streamlit_app/ is importable
when running from the ``frontend/`` directory.

Run locally:
    cd frontend
    streamlit run main.py

Or from the repo root (original behaviour):
    streamlit run streamlit_app/app.py
"""

import os
import sys

# Ensure the monorepo root is on sys.path so streamlit_app package imports work.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Re-run the actual Streamlit application file.
# Streamlit's --module-execution model means we just exec() it.
exec(open(os.path.join(REPO_ROOT, "streamlit_app", "app.py")).read())
