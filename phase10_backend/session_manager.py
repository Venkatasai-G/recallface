import uuid

from .database import get_connection


# ============================================================
# CREATE SESSION
# ============================================================

def create_session():
    """
    Create a new RecallFace session.

    Returns:
        str: Unique session ID
    """

    session_id = str(uuid.uuid4())

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO sessions (session_id)
        VALUES (?)
        """,
        (session_id,),
    )

    connection.commit()

    connection.close()

    return session_id


# ============================================================
# SAVE ROUND
# ============================================================

def save_round(
    session_id,
    round_number,
    selected_face,
    selected_image_path,
    confidence,
    guidance,
):
    """
    Save one completed RecallFace round.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO rounds (
            session_id,
            round_number,
            selected_face,
            selected_image_path,
            confidence,
            guidance
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            round_number,
            selected_face,
            selected_image_path,
            confidence,
            guidance,
        ),
    )

    connection.commit()

    connection.close()


# ============================================================
# GET SESSION ROUNDS
# ============================================================

def get_session_rounds(session_id):
    """
    Retrieve all saved rounds belonging to a session.

    Returns:
        list[dict]: Saved round information
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            round_number,
            selected_face,
            selected_image_path,
            confidence,
            guidance,
            created_at
        FROM rounds
        WHERE session_id = ?
        ORDER BY round_number ASC
        """,
        (session_id,),
    )

    rows = cursor.fetchall()

    connection.close()

    return [dict(row) for row in rows]


# ============================================================
# GET SESSION
# ============================================================

def get_session(session_id):
    """
    Retrieve basic information about a session.

    Returns:
        dict | None
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            session_id,
            status,
            created_at,
            completed_at
        FROM sessions
        WHERE session_id = ?
        """,
        (session_id,),
    )

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return None

    return dict(row)

# ============================================================
# FINALIZE SESSION
# ============================================================

def finalize_session(session_id):
    """
    Mark a RecallFace session as completed.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE sessions
        SET
            status = 'completed',
            completed_at = CURRENT_TIMESTAMP
        WHERE session_id = ?
        """,
        (session_id,),
    )

    connection.commit()

    connection.close()