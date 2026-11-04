"""Optional scheduled cleanup: python -m app.cleanup_history."""
from . import config
from .database import from_env
from .history import cleanup


def main():
    database = from_env()
    if database is None:
        raise SystemExit('Set DATABASE_URL before running history cleanup.')
    try:
        cleanup(database)
        print('Expired browser sessions and their history removed.')
    finally:
        database.close()


if __name__ == '__main__':
    main()
