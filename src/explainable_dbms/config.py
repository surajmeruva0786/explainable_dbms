"""
Configuration utilities for the Explainable DBMS pipeline.

Environment variables are used when available, otherwise sensible defaults
are provided for local development. Update the defaults or set environment
variables before running the pipeline in production.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Connection settings for the database (MySQL or SQLite)."""

    db_type: str = os.getenv("DB_TYPE", "sqlite")  # "mysql" or "sqlite"
    user: str = os.getenv("MYSQL_USER", "root")
    password: str = os.getenv("MYSQL_PASSWORD", "password")
    host: str = os.getenv("MYSQL_HOST", "127.0.0.1")
    port: int = int(os.getenv("MYSQL_PORT", 3306))
    database: str = os.getenv("MYSQL_DATABASE", "explainable_dbms")
    sqlite_path: Path = Path(os.getenv("SQLITE_PATH", "explainable_dbms.db"))

    def build_url(self, include_database: bool = True) -> str:
        """Return a SQLAlchemy-compatible database URL."""
        if self.db_type.lower() == "sqlite":
            return f"sqlite:///{self.sqlite_path}"
        else:
            # MySQL connection
            db_segment = f"/{self.database}" if include_database else ""
            return (
                f"mysql+mysqlconnector://{self.user}:{self.password}"
                f"@{self.host}:{self.port}{db_segment}"
            )


@dataclass
class PathsConfig:
    """Filesystem locations for outputs and artifacts."""

    project_root: Path = Path(os.getenv("PROJECT_ROOT", Path.cwd()))
    output_dir: Path = project_root / "outputs"
    visualizations_dir: Path = output_dir / "visualizations"
    explanations_dir: Path = output_dir / "explanations"
    metrics_path: Path = output_dir / "benchmark_results.json"

    def ensure_directories(self) -> None:
        """Create any directories that do not yet exist."""
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        self.explanations_dir.mkdir(parents=True, exist_ok=True)


def get_database_config() -> DatabaseConfig:
    """Factory for database configuration."""
    return DatabaseConfig()


def get_paths_config() -> PathsConfig:
    """Factory for path configuration with directories guaranteed to exist."""
    paths = PathsConfig()
    paths.ensure_directories()
    return paths

