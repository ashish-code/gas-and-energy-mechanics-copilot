"""
Tests: tests/test_memory.py

PURPOSE:
    Unit tests for the DynamoDB conversation memory manager.
    These tests use mocking to avoid requiring real AWS credentials, making
    them runnable in CI without infrastructure.

    We test:
      1. Happy path: add turns, retrieve them, format as context.
      2. DynamoDB failure: graceful degradation (no crash, empty results).
      3. Session clearing: batch delete works correctly.
      4. ConversationTurn serialisation: DynamoDB item round-trip.

TESTING PHILOSOPHY:
    Memory is stateful infrastructure — its correctness is critical for
    conversation continuity. We test all public methods with:
      - Success paths (DynamoDB responds normally)
      - Failure paths (DynamoDB raises ClientError, no credentials)
      - Edge cases (empty history, very long content, TTL calculation)

WHY MOCK DynamoDB:
    Unit tests should be fast and not require external services. The boto3
    `moto` library provides an in-memory DynamoDB emulator that is ideal
    for testing. We use unittest.mock here to keep dependencies minimal
    (moto would require an additional dev dependency).

    Alternative: moto library (`pip install moto[dynamodb]`) — provides
    full DynamoDB emulation including Query, PutItem, BatchWriter.
    Preferred for more integration-level tests.
"""

from __future__ import annotations

import os
import time
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Set CONFIG_DIR before importing application modules
os.environ.setdefault("CONFIG_DIR", "./config")


class TestConversationTurn:
    """Tests for the ConversationTurn dataclass serialisation."""

    def test_to_dynamodb_item_has_required_keys(self):
        """DynamoDB PutItem items must have the exact partition and sort keys."""
        from gas_energy_copilot.ai_copilot.services.memory import ConversationTurn

        turn = ConversationTurn(
            session_id="sess-123",
            turn_id="0001",
            role="user",
            content="What is MAOP?",
        )
        item = turn.to_dynamodb_item(ttl_seconds=9999999)

        assert item["session_id"] == "sess-123"
        assert item["turn_id"] == "0001"
        assert item["role"] == "user"
        assert item["content"] == "What is MAOP?"
        assert item["ttl"] == 9999999
        assert "timestamp" in item
        assert "metadata" in item

    def test_from_dynamodb_item_round_trip(self):
        """Items written to DynamoDB should deserialise back to identical ConversationTurn."""
        from gas_energy_copilot.ai_copilot.services.memory import ConversationTurn

        original = ConversationTurn(
            session_id="sess-456",
            turn_id="0003",
            role="assistant",
            content="MAOP stands for Maximum Allowable Operating Pressure.",
            metadata={"judge_confidence": 0.92},
        )
        item = original.to_dynamodb_item(ttl_seconds=1234567890)
        restored = ConversationTurn.from_dynamodb_item(item)

        assert restored.session_id == original.session_id
        assert restored.turn_id == original.turn_id
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.metadata == original.metadata

    def test_empty_metadata_defaults_to_dict(self):
        """ConversationTurn should never have None metadata (DynamoDB rejects None for Map)."""
        from gas_energy_copilot.ai_copilot.services.memory import ConversationTurn

        turn = ConversationTurn(session_id="s", turn_id="0001", role="user", content="q")
        item = turn.to_dynamodb_item(ttl_seconds=0)
        assert item["metadata"] == {}


class TestDynamoDBMemoryManagerMocked:
    """
    Tests for DynamoDBMemoryManager with mocked boto3.

    Each test creates a fresh manager instance and patches the DynamoDB resource
    at the boto3 level to avoid real AWS calls.
    """

    def _make_manager(self):
        """Helper: create a DynamoDBMemoryManager with a fresh config."""
        from gas_energy_copilot.ai_copilot.services.memory import DynamoDBMemoryManager
        return DynamoDBMemoryManager()

    def test_get_history_returns_chronological_order(self):
        """
        get_history should return turns sorted oldest-first (chronological).

        DynamoDB Query with ScanIndexForward=False returns newest first.
        The manager reverses this to restore chronological order.
        """
        from gas_energy_copilot.ai_copilot.services.memory import (
            DynamoDBMemoryManager,
            ConversationTurn,
        )

        # Mock DynamoDB items (newest first, as DynamoDB returns them with ScanIndexForward=False)
        mock_items = [
            {"session_id": "s", "turn_id": "0003", "role": "assistant",
             "content": "Answer 2", "timestamp": "2024-01-01T10:02:00Z", "metadata": {}},
            {"session_id": "s", "turn_id": "0002", "role": "user",
             "content": "Question 2", "timestamp": "2024-01-01T10:01:00Z", "metadata": {}},
            {"session_id": "s", "turn_id": "0001", "role": "user",
             "content": "Question 1", "timestamp": "2024-01-01T10:00:00Z", "metadata": {}},
        ]

        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": mock_items}

        manager = self._make_manager()
        manager._table = mock_table  # inject mock table directly

        turns = manager.get_history("s")

        assert len(turns) == 3
        # Chronological order: 0001, 0002, 0003
        assert turns[0].turn_id == "0001"
        assert turns[1].turn_id == "0002"
        assert turns[2].turn_id == "0003"

    def test_get_history_returns_empty_on_dynamodb_error(self):
        """
        Graceful degradation: DynamoDB errors should return empty list, not raise.

        This test verifies that the memory system doesn't crash the chat endpoint
        when DynamoDB is unreachable (e.g., during network issues or cold start).
        """
        from botocore.exceptions import ClientError
        from gas_energy_copilot.ai_copilot.services.memory import DynamoDBMemoryManager

        mock_table = MagicMock()
        mock_table.query.side_effect = ClientError(
            {"Error": {"Code": "ProvisionedThroughputExceededException", "Message": "throttled"}},
            "Query",
        )

        manager = self._make_manager()
        manager._table = mock_table

        result = manager.get_history("session-xyz")
        assert result == []  # empty, not an exception

    def test_add_turn_uses_next_sequential_turn_id(self):
        """
        add_turn should auto-assign turn_id as zero-padded count + 1.

        If 3 turns exist, the next turn_id should be "0004".
        """
        from gas_energy_copilot.ai_copilot.services.memory import DynamoDBMemoryManager

        mock_table = MagicMock()
        mock_table.query.return_value = {"Count": 3}  # 3 existing turns
        mock_table.put_item.return_value = {}

        manager = self._make_manager()
        manager._table = mock_table

        result = manager.add_turn("session-abc", "user", "What is §192.505?")

        assert result is True
        # Check that put_item was called with the correct turn_id
        call_args = mock_table.put_item.call_args[1]  # keyword args
        item = call_args["Item"]
        assert item["turn_id"] == "0004"
        assert item["role"] == "user"

    def test_add_turn_returns_false_on_error(self):
        """add_turn should return False on DynamoDB failure, not raise."""
        from botocore.exceptions import ClientError
        from gas_energy_copilot.ai_copilot.services.memory import DynamoDBMemoryManager

        mock_table = MagicMock()
        mock_table.query.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Table not found"}},
            "Query",
        )

        manager = self._make_manager()
        manager._table = mock_table

        result = manager.add_turn("session-err", "user", "test question")
        assert result is False

    def test_format_as_context_empty_history(self):
        """format_as_context with no turns should return empty string."""
        from gas_energy_copilot.ai_copilot.services.memory import DynamoDBMemoryManager

        manager = self._make_manager()
        result = manager.format_as_context([])
        assert result == ""

    def test_format_as_context_includes_all_turns(self):
        """format_as_context should include User/Assistant prefixes for all turns."""
        from gas_energy_copilot.ai_copilot.services.memory import (
            DynamoDBMemoryManager,
            ConversationTurn,
        )

        manager = self._make_manager()
        turns = [
            ConversationTurn("s", "0001", "user", "What is MAOP?"),
            ConversationTurn("s", "0002", "assistant", "MAOP is the Maximum Allowable..."),
        ]
        result = manager.format_as_context(turns)

        assert "CONVERSATION HISTORY:" in result
        assert "User: What is MAOP?" in result
        assert "Assistant: MAOP is the Maximum Allowable..." in result

    def test_format_as_context_truncates_long_content(self):
        """format_as_context should truncate content longer than 500 chars."""
        from gas_energy_copilot.ai_copilot.services.memory import (
            DynamoDBMemoryManager,
            ConversationTurn,
        )

        long_content = "A" * 1000  # 1000 chars, should be truncated to 500 + "..."
        manager = self._make_manager()
        turns = [ConversationTurn("s", "0001", "user", long_content)]
        result = manager.format_as_context(turns)

        assert "..." in result  # truncation marker present
        # The result line should not contain the full 1000-char string
        assert "A" * 501 not in result

    def test_get_table_returns_none_when_no_credentials(self):
        """
        _get_table should return None (not raise) when AWS credentials are absent.
        Graceful degradation is critical for local dev environments.
        """
        from botocore.exceptions import NoCredentialsError
        from gas_energy_copilot.ai_copilot.services.memory import DynamoDBMemoryManager

        manager = self._make_manager()

        with patch("boto3.resource") as mock_resource:
            mock_resource.side_effect = NoCredentialsError()
            table = manager._get_table()

        assert table is None

    def test_clear_session_deletes_all_turns(self):
        """clear_session should delete all items for a session."""
        from gas_energy_copilot.ai_copilot.services.memory import DynamoDBMemoryManager

        mock_items = [
            {"session_id": "s", "turn_id": "0001"},
            {"session_id": "s", "turn_id": "0002"},
        ]
        mock_batch_writer = MagicMock()
        mock_batch_writer.__enter__ = MagicMock(return_value=mock_batch_writer)
        mock_batch_writer.__exit__ = MagicMock(return_value=False)

        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": mock_items}
        mock_table.batch_writer.return_value = mock_batch_writer

        manager = self._make_manager()
        manager._table = mock_table

        deleted = manager.clear_session("s")

        assert deleted == 2
        assert mock_batch_writer.delete_item.call_count == 2


class TestGetMemoryManagerSingleton:
    """Tests for the module-level singleton function."""

    def test_returns_same_instance(self):
        """get_memory_manager() must return the same object on repeated calls."""
        from gas_energy_copilot.ai_copilot.services.memory import get_memory_manager

        mgr1 = get_memory_manager()
        mgr2 = get_memory_manager()
        assert mgr1 is mgr2
