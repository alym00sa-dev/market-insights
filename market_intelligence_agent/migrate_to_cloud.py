"""
Migrate local ArangoDB → ArangoDB Cloud.

Usage:
  PYTHONPATH=. python migrate_to_cloud.py --cloud-host https://abc123.arangodb.cloud:8529 \
                                           --cloud-password YOUR_PASSWORD

The script reads from localhost:8529 and writes to the cloud instance.
Safe to re-run — skips documents that already exist.
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arango import ArangoClient


LOCAL_HOST = "http://localhost:8529"
LOCAL_USER = "root"
LOCAL_PASSWORD = "password"
LOCAL_DB = "market_intel"

CLOUD_USER = "root"
CLOUD_DB = "market_intel"


def connect(host: str, user: str, password: str, db_name: str):
    client = ArangoClient(hosts=host)
    try:
        sys_db = client.db("_system", username=user, password=password)
        if not sys_db.has_database(db_name):
            sys_db.create_database(db_name)
            print(f"  Created database: {db_name}")
    except Exception:
        pass  # AuraDB: pre-created, restricted _system access
    return client.db(db_name, username=user, password=password)


def migrate_collection(local_db, cloud_db, collection_name: str, is_edge: bool = False):
    if not cloud_db.has_collection(collection_name):
        cloud_db.create_collection(collection_name, edge=is_edge)
        print(f"  Created collection: {collection_name}")

    local_col = local_db.collection(collection_name)
    cloud_col = cloud_db.collection(collection_name)

    docs = list(local_col.all())
    inserted = 0
    skipped = 0

    for doc in docs:
        key = doc["_key"]
        if cloud_col.has(key):
            skipped += 1
            continue
        # Strip internal ArangoDB fields that will be set on insert
        clean = {k: v for k, v in doc.items() if k not in ("_id", "_rev")}
        try:
            cloud_col.insert(clean)
            inserted += 1
        except Exception as e:
            print(f"    Warning: could not insert {key}: {e}")

    print(f"  {collection_name}: {inserted} inserted, {skipped} already existed (total local: {len(docs)})")
    return inserted


def run(cloud_host: str, cloud_password: str):
    print(f"\n=== Migrating local ArangoDB → Cloud ===")
    print(f"  From: {LOCAL_HOST}/{LOCAL_DB}")
    print(f"  To:   {cloud_host}/{CLOUD_DB}\n")

    print("Connecting to local...")
    local_db = connect(LOCAL_HOST, LOCAL_USER, LOCAL_PASSWORD, LOCAL_DB)
    print("Connected to local.\n")

    print("Connecting to cloud...")
    cloud_db = connect(cloud_host, CLOUD_USER, cloud_password, CLOUD_DB)
    print("Connected to cloud.\n")

    # Migrate in dependency order: players first, then events, then edges
    print("Migrating ai_players...")
    migrate_collection(local_db, cloud_db, "ai_players", is_edge=False)

    print("\nMigrating events...")
    migrate_collection(local_db, cloud_db, "events", is_edge=False)

    print("\nMigrating player_events (edges)...")
    migrate_collection(local_db, cloud_db, "player_events", is_edge=True)

    # Ensure graph definition exists on cloud
    print("\nEnsuring graph definition...")
    if not cloud_db.has_graph("market_intel"):
        cloud_db.create_graph(
            "market_intel",
            edge_definitions=[{
                "edge_collection": "player_events",
                "from_vertex_collections": ["ai_players"],
                "to_vertex_collections": ["events"],
            }],
        )
        print("  Created graph: market_intel")
    else:
        print("  Graph already exists.")

    # Verify
    total_events = cloud_db.collection("events").count()
    total_players = cloud_db.collection("ai_players").count()
    total_edges = cloud_db.collection("player_events").count()
    print(f"\n=== Done ===")
    print(f"  Cloud DB now has:")
    print(f"    {total_players} players")
    print(f"    {total_events} events")
    print(f"    {total_edges} edges")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud-host", required=True, help="e.g. https://abc123.arangodb.cloud:8529")
    parser.add_argument("--cloud-password", required=True, help="AuraDB root password")
    args = parser.parse_args()
    run(args.cloud_host, args.cloud_password)
