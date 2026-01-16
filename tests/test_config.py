
import pytest
import os
from pathlib import Path
from config import DeviceConfig, ModelConfig, GenerationConfig

def test_device_config_defaults():
    config = DeviceConfig()
    assert config.device_type == "cpu"  # Default in class definition
    assert config.attention_slice_size == 1

def test_model_config_defaults():
    config = ModelConfig()
    assert config.enable_face_restore is True
    assert config.codeformer_fidelity == 0.7

def test_generation_config_defaults():
    config = GenerationConfig()
    assert config.strength == 0.75
    assert config.guidance_scale == 7.0
    assert config.width == 512
    assert config.height == 512

def test_paths_exist():
    from config import MODELS_DIR, OUTPUTS_DIR, DATASETS_DIR
    assert MODELS_DIR.exists()
    assert OUTPUTS_DIR.exists()
    assert DATASETS_DIR.exists()
