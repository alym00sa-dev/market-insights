"""
Neo4j schema initialization and player seed data.
Run once on first startup; safe to re-run (idempotent).
"""
import yaml
from pathlib import Path
from graph.client import run_query

PLAYERS_YAML = Path(__file__).parent.parent / "config" / "players.yaml"


def initialize():
    _ensure_constraints()
    _seed_players()
    print("Schema initialized.")


def _ensure_constraints():
    run_query("CREATE CONSTRAINT player_key IF NOT EXISTS FOR (p:Player) REQUIRE p.key IS UNIQUE")
    run_query("CREATE CONSTRAINT event_hash IF NOT EXISTS FOR (e:Event) REQUIRE e.raw_content_hash IS UNIQUE")
    run_query("CREATE INDEX event_published IF NOT EXISTS FOR (e:Event) ON (e.published_date)")
    run_query("CREATE INDEX event_type_idx IF NOT EXISTS FOR (e:Event) ON (e.event_type)")
    run_query("CREATE INDEX event_first_seen IF NOT EXISTS FOR (e:Event) ON (e.first_seen)")
    print("  Constraints and indexes ensured.")


def _seed_players():
    with open(PLAYERS_YAML) as f:
        data = yaml.safe_load(f)

    for player in data["players"]:
        run_query(
            "MERGE (p:Player {key: $key}) ON CREATE SET p += $props",
            {"key": player["key"], "props": player},
        )
        print(f"  Seeded player: {player['name']}")


if __name__ == "__main__":
    initialize()
