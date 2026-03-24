"""
Quick ArangoDB connection test.
Run with: PYTHONPATH=. python test_db_connection.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings

print(f"Connecting to: {settings.ARANGO_HOST}")
print(f"Database:      {settings.ARANGO_DB}")
print(f"User:          {settings.ARANGO_USER}")
print()

try:
    from graph.client import get_db
    db = get_db()
    collections = [c["name"] for c in db.collections() if not c["name"].startswith("_")]
    print(f"Connected successfully.")
    print(f"Collections: {collections or '(empty — run schema init next)'}")
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)
