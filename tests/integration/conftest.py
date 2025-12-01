"""
Pytest configuration for integration tests.
"""

import pytest


@pytest.fixture(scope="session")
def register_tools():
    """
    Register all default tools for integration tests.
    
    Only loads tools when explicitly requested by tests.
    """
    try:
        from src.agent.tools import register_default_tools
        register_default_tools()
    except ImportError as e:
        pytest.skip(f"Tool dependencies not available: {e}")
    yield