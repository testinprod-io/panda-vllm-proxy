import pytest
from app.config import get_settings   # adjust the import path

@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """
    Ensure each test starts with a fresh Settings() object.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()