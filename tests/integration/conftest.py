"""
Pytest configuration for integration tests.
"""

import pytest
from src.agent.tools import register_default_tools


@pytest.fixture(scope="session", autouse=True)
def register_tools():
    """Register all default tools for integration tests."""
    register_default_tools()
    yield