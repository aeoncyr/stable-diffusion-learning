
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_ui_imports():
    """Test that UI modules can be imported without error"""
    # Mock heavy dependencies
    with patch.dict(sys.modules, {
        "torch": MagicMock(),
        "diffusers": MagicMock(),
        "diffusers.utils": MagicMock(), # Vital for imports
        "gradio": MagicMock(),
        "psutil": MagicMock(),
        "PIL": MagicMock(),
        "numpy": MagicMock(),
        "cv2": MagicMock(),
        "torchvision": MagicMock(),
        "torchvision.transforms": MagicMock(),
        "torchvision.transforms.functional": MagicMock(), 
    }):
        # Mock gradio.Blocks context manager
        sys.modules["gradio"].Blocks.return_value.__enter__.return_value = MagicMock()
        
        try:
            from ui import memory_monitor
            from ui import editor_tab
            import app
        except ImportError as e:
            pytest.fail(f"Failed to import UI modules: {e}")

def test_memory_monitor_creation():
    """Test memory monitor component creation"""
    with patch.dict(sys.modules, {
        "gradio": MagicMock(),
        "core.memory_manager": MagicMock(),
    }):
        from ui import memory_monitor
        
        # Test creation
        comp, timer = memory_monitor.create_memory_monitor()
        assert comp is not None
        assert timer is not None
