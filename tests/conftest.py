
import pytest
import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("SD_MODELS_DIR", "test_models")
    monkeypatch.setenv("SD_OUTPUTS_DIR", "test_outputs")
    monkeypatch.setenv("SD_DATASETS_DIR", "test_datasets")
