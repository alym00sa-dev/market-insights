import time
from neo4j import GraphDatabase
from config.settings import settings

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    return _driver


def run_query(cypher: str, params: dict = None) -> list[dict]:
    driver = get_driver()
    with driver.session(database=settings.NEO4J_DATABASE) as session:
        result = session.run(cypher, params or {})
        return [record.data() for record in result]


def wait_for_connection(max_wait: int = 120, interval: int = 10) -> bool:
    """
    Retry Neo4j connection until reachable or max_wait seconds have elapsed.
    AuraDB free tier pauses after inactivity and needs a moment to resume
    once you hit Resume in the console — this polls until it's back.
    """
    global _driver
    deadline = time.time() + max_wait
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            _driver = None  # force fresh driver on each attempt
            run_query("RETURN 1 AS ok")
            if attempt > 1:
                print(f"  [Neo4j] Connected after {attempt} attempt(s).")
            else:
                print("  [Neo4j] Connection OK.")
            return True
        except Exception as e:
            remaining = max(0, int(deadline - time.time()))
            print(f"  [Neo4j] Not reachable (attempt {attempt}, {remaining}s remaining): {type(e).__name__}")
            if remaining > 0:
                time.sleep(interval)
    print("\n  Neo4j could not be reached. If using AuraDB free tier:")
    print("  → Resume your instance at https://console.neo4j.io")
    print("  → Then re-run this script.\n")
    return False
