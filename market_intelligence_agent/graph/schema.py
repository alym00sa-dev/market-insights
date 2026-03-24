"""
ArangoDB schema initialization and player seed data.
Run once on first startup; safe to re-run (idempotent).
"""
import yaml
from pathlib import Path
from graph.client import get_db

PLAYERS_YAML = Path(__file__).parent.parent / "config" / "players.yaml"


def initialize(db=None):
    if db is None:
        db = get_db()

    _ensure_collections(db)
    _ensure_graph(db)
    _seed_players(db)
    print("Schema initialized.")


def _ensure_collections(db):
    if not db.has_collection("ai_players"):
        db.create_collection("ai_players")

    if not db.has_collection("events"):
        col = db.create_collection("events")
        col.add_index({"type": "persistent", "fields": ["raw_content_hash"], "unique": True, "sparse": True})
        col.add_index({"type": "persistent", "fields": ["published_date"]})
        col.add_index({"type": "persistent", "fields": ["event_type"]})
        col.add_index({"type": "persistent", "fields": ["scraped_date"]})

    if not db.has_collection("player_events"):
        db.create_collection("player_events", edge=True)


def _ensure_graph(db):
    if not db.has_graph("market_intel"):
        db.create_graph(
            "market_intel",
            edge_definitions=[
                {
                    "edge_collection": "player_events",
                    "from_vertex_collections": ["ai_players"],
                    "to_vertex_collections": ["events"],
                }
            ],
        )


def _seed_players(db):
    collection = db.collection("ai_players")
    with open(PLAYERS_YAML) as f:
        data = yaml.safe_load(f)

    for player in data["players"]:
        key = player["key"]
        if not collection.has(key):
            collection.insert({**player, "_key": key})
            print(f"  Seeded player: {player['name']}")
        else:
            print(f"  Player already exists: {player['name']}")


if __name__ == "__main__":
    initialize()
