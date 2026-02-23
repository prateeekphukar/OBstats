"""Test configuration for pytest-asyncio."""

import pytest


# Enable asyncio mode for all tests
@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
