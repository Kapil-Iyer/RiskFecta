"""
RiskFecta Phase 1B — PostgreSQL (Supabase-hosted) connectivity + schema
application.

Scope (locked, see BUILD_PLAN.md Phase 1):
- Supabase is used as managed PostgreSQL only, via the standard
  `DATABASE_URL` connection string (TRD.md §5). No Supabase client library,
  Auth, Storage, Edge Functions, or Realtime dependency is introduced here.
- Credentials are never logged, printed, or embedded in an exception message
  raised by this module — only `config.get_database_url()` ever sees the
  connection string, and it is passed straight to psycopg2.
- `schema.sql` is the sole source of DDL. This module applies it verbatim
  (`CREATE TABLE IF NOT EXISTS`, so re-applying is a no-op against an
  up-to-date database) and never invents tables/columns of its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2

import config

SCHEMA_PATH = config.PROJECT_ROOT / "schema.sql"


def get_connection():
    """Open a new psycopg2 connection using DATABASE_URL from the environment.

    Never logs or embeds the connection string; callers are responsible for
    closing the returned connection (or use it as a context manager).
    """
    return psycopg2.connect(config.get_database_url())


def apply_schema(conn=None, schema_path: Optional[Path] = None) -> None:
    """Apply schema.sql to the configured database.

    Uses `CREATE TABLE IF NOT EXISTS` (as written in schema.sql), so this is
    safe to re-run against an already-migrated database. Runs inside a single
    transaction: any failure leaves the schema untouched.
    """
    owns_conn = conn is None
    conn = conn or get_connection()
    path = schema_path or SCHEMA_PATH
    sql = path.read_text()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
    finally:
        if owns_conn:
            conn.close()


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    is_nullable: bool


def table_exists(conn, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = %s)",
            (table,),
        )
        return bool(cur.fetchone()[0])


def list_columns(conn, table: str) -> List[ColumnInfo]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position",
            (table,),
        )
        rows = cur.fetchall()
    return [ColumnInfo(name=r[0], data_type=r[1], is_nullable=(r[2] == "YES")) for r in rows]


def unique_constraint_columns(conn, table: str) -> List[List[str]]:
    """Return the column-name lists for every UNIQUE constraint on `table`
    (each inner list is one constraint's columns, in key order)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT tc.constraint_name, kcu.column_name, kcu.ordinal_position
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
            WHERE tc.table_schema = 'public' AND tc.table_name = %s AND tc.constraint_type = 'UNIQUE'
            ORDER BY tc.constraint_name, kcu.ordinal_position
            """,
            (table,),
        )
        rows = cur.fetchall()
    constraints: Dict[str, List[str]] = {}
    for name, col, _ in rows:
        constraints.setdefault(name, []).append(col)
    return list(constraints.values())


def fetch_scalar(conn, sql: str, params: Optional[tuple] = None) -> Any:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else None
