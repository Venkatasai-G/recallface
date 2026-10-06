import sqlite3
from pathlib import Path


# ============================================================
# DATABASE PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_DIR = PROJECT_ROOT / "data" / "database"

DATABASE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATABASE_PATH = DATABASE_DIR / "recallface.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """
    Create and return a connection to the RecallFace SQLite database.
    """

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():
    """
    Create the RecallFace database tables if they do not exist.
    """

    connection = get_connection()

    cursor = connection.cursor()

    # --------------------------------------------------------
    # SESSIONS TABLE
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
        """
    )

    # --------------------------------------------------------
    # ROUNDS TABLE
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS rounds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            round_number INTEGER NOT NULL,
            selected_face INTEGER,
            selected_image_path TEXT,
            confidence INTEGER,
            guidance TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (session_id)
                REFERENCES sessions(session_id)
        )
        """
    )

    connection.commit()

    connection.close()

# ============================================================
# DATABASE MIGRATION
# ============================================================

def migrate_database():
    """
    Add new session columns to an existing RecallFace database.
    """

    connection = get_connection()

    cursor = connection.cursor()

    # Get existing columns from sessions table
    cursor.execute(
        """
        PRAGMA table_info(sessions)
        """
    )

    existing_columns = {
        row["name"]
        for row in cursor.fetchall()
    }

    # Add status column if it does not exist
    if "status" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE sessions
            ADD COLUMN status TEXT NOT NULL DEFAULT 'active'
            """
        )

    # Add completed_at column if it does not exist
    if "completed_at" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE sessions
            ADD COLUMN completed_at TIMESTAMP
            """
        )

    connection.commit()

    connection.close()


# ============================================================
# INITIALIZE DATABASE
# ============================================================

if __name__ == "__main__":

    initialize_database()

    migrate_database()

    print(
        f"RecallFace database initialized at:\n"
        f"{DATABASE_PATH}"
    )