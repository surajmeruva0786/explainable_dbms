"""
Part 1: Database setup and schema creation.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

from .config import DatabaseConfig, get_database_config
from .models import Base


def create_mysql_database(config: DatabaseConfig) -> None:
    """
    Ensure the database defined in ``config`` exists.

    This uses a server-level connection (no database name in the URL) to
    execute ``CREATE DATABASE IF NOT EXISTS`` safely.
    """
    server_engine = create_engine(config.build_url(include_database=False), isolation_level="AUTOCOMMIT")
    with server_engine.connect() as connection:
        connection.execute(text(f"CREATE DATABASE IF NOT EXISTS {config.database}"))


def get_engine(config: DatabaseConfig | None = None) -> Engine:
    """Return a SQLAlchemy engine bound to the configured database."""
    cfg = config or get_database_config()
    return create_engine(cfg.build_url(), echo=False, pool_pre_ping=True)


def create_tables(engine: Engine) -> None:
    """Create all tables defined in the ORM models."""
    Base.metadata.create_all(engine)


def drop_tables(engine: Engine) -> None:
    """Utility to drop all tables (useful for development resets)."""
    try:
        Base.metadata.drop_all(engine)
    except ProgrammingError as exc:
        raise RuntimeError("Failed to drop tables. Check database permissions.") from exc


def drop_specific_table(engine: Engine, table_name: str) -> None:
    """Utility to drop a specific table."""
    try:
        with engine.connect() as connection:
            connection.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            connection.commit()
    except ProgrammingError as exc:
        raise RuntimeError(f"Failed to drop table {table_name}. Check database permissions.") from exc

def initialize_database(config: DatabaseConfig | None = None) -> Engine:
    """
    Top-level helper to create the database (if required) and return an engine.
    """
    cfg = config or get_database_config()
    create_mysql_database(cfg)
    engine = get_engine(cfg)
    drop_specific_table(engine, "prediction_results")
    create_tables(engine)
    return engine


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Produce a configured session factory bound to the engine."""
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """
    Provide a transactional scope for DB operations.
    """
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

