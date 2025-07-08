# tests/test_router.py
import pytest
from unittest.mock import patch, MagicMock
from langgraph_flow import router

@patch('psycopg2.connect')
def test_router_happy_path_single_course(mock_connect):
    """
    Tests the router's happy path where a user has exactly one syllabus,
    so the choice is unambiguous.
    """
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        (101, 'CS101', 'Intro to CS', 'Dr. Turing', 'Fall 2025')
    ]
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    initial_state = {"user_id": "test_user", "query": "When is the final?"}
    result = router(initial_state)

    assert result["syllabus_id"] == 101
    assert result["_next"] == "load_context"
