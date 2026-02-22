# app/database.py
# CHANGE: 起動時に簡易マイグレーション（ALTER TABLE）で body_fat 等の列欠落を自動修復
# CHANGE: SQLite 接続設定の安定化

from __future__ import annotations

import os
import sqlite3
from typing import Dict, List, Tuple

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DB_PATH = os.getenv("PHYSIQUE_DB_PATH", "physique.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

Base = declarative_base()


def _get_sqlite_columns(conn: sqlite3.Connection, table_name: str) -> Dict[str, str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    rows = cur.fetchall()
    # rows: (cid, name, type, notnull, dflt_value, pk)
    return {r[1]: (r[2] or "").upper() for r in rows}


def _sqlite_add_column_if_missing(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    col_type_sql: str,
) -> bool:
    cols = _get_sqlite_columns(conn, table)
    if column in cols:
        return False
    cur = conn.cursor()
    cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type_sql}")
    conn.commit()
    return True


def ensure_sqlite_schema() -> None:
    """
    SQLAlchemy の create_all は既存テーブルへの列追加をしないため、
    body_fat 追加後などに SQLite schema 不整合が起きる。
    ここで最低限の ALTER TABLE を実施し、起動時に整合を取る。
    """
    if not os.path.exists(DB_PATH):
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        # WeightRecord テーブルがある場合のみ対応
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='weight_records'"
        )
        exists = cur.fetchone()
        if not exists:
            return

        # 必須追加列（将来増えてもここに足せばOK）
        _sqlite_add_column_if_missing(conn, "weight_records", "body_fat", "REAL")
        _sqlite_add_column_if_missing(conn, "weight_records", "calorie", "REAL")
        # created_at は既存設計である前提（欠落は作り直し推奨）
    finally:
        conn.close()


def init_db() -> None:
    # まずテーブル作成（なければ作る）
    Base.metadata.create_all(bind=engine)

    # 既存テーブルの列整合（body_fat等）
    ensure_sqlite_schema()