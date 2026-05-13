"""
Quick Neo4j connection test.
Run with: PYTHONPATH=. python scripts/test_db_connection.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings

print(f"Connecting to: {settings.NEO4J_URI}")
print(f"Database:      {settings.NEO4J_DATABASE}")
print(f"User:          {settings.NEO4J_USER}")
print()

from graph.client import wait_for_connection, run_query

if not wait_for_connection(max_wait=120, interval=10):
    sys.exit(1)

results = run_query(
    "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY label"
)
if results:
    for r in results:
        print(f"  {r['label']}: {r['count']} nodes")
else:
    print("  Graph is empty — run schema init next.")
