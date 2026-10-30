import sqlite3
from pathlib import Path
from typing import Optional, Tuple

DB_PATH = Path(__file__).resolve().parent.parent / "agriagent.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT,
                role TEXT DEFAULT 'user',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                file_name TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                summary TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def ensure_default_user() -> None:
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT 1 FROM users WHERE username = 'admin'"
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)",
                ("admin", "admin123", "admin@agriagent.local", "admin"),
            )
            conn.commit()
        else:
            conn.execute(
                "UPDATE users SET role = 'admin' WHERE username = 'admin' AND (role IS NULL OR role = '')",
            )
            conn.commit()
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> Optional[str]:
    conn = get_connection()
    try:
        user = conn.execute(
            "SELECT username FROM users WHERE username = ? AND password = ?",
            (username.strip(), password),
        ).fetchone()
        return user["username"] if user else None
    finally:
        conn.close()


def register_user(username: str, password: str, email: Optional[str] = None) -> Tuple[bool, str]:
    username = username.strip()
    if not username or not password:
        return False, "Username and password are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT 1 FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if existing:
            return False, "Username already exists."
        conn.execute(
            "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
            (username, password, email or ""),
        )
        conn.commit()
        return True, "User created successfully."
    finally:
        conn.close()


def save_session(username: str, file_name: str, summary: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO sessions (username, file_name, summary) VALUES (?, ?, ?)",
            (username, file_name, summary),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_sessions(username: str, limit: int = 5):
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT file_name, created_at, summary FROM sessions WHERE username = ? ORDER BY id DESC LIMIT ?",
            (username, limit),
        ).fetchall()
    finally:
        conn.close()
