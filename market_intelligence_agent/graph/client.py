from arango import ArangoClient
from config.settings import settings

_db = None


def get_db():
    global _db
    if _db is None:
        client = ArangoClient(hosts=settings.ARANGO_HOST)
        # Try to create the DB if it doesn't exist.
        # On ArangoDB Cloud (AuraDB), the DB is pre-created and _system
        # access may be restricted — in that case we skip creation silently.
        try:
            sys_db = client.db(
                "_system",
                username=settings.ARANGO_USER,
                password=settings.ARANGO_PASSWORD,
            )
            if not sys_db.has_database(settings.ARANGO_DB):
                sys_db.create_database(settings.ARANGO_DB)
        except Exception:
            pass  # AuraDB: DB pre-exists, no _system access needed
        _db = client.db(
            settings.ARANGO_DB,
            username=settings.ARANGO_USER,
            password=settings.ARANGO_PASSWORD,
        )
    return _db
