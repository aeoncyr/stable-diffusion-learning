
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

# Ensure project root is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_edit_image_flow():
    # Mock heavy dependencies in sys.modules BEFORE import
    with patch.dict(sys.modules, {
        "torch": MagicMock(),
        "diffusers": MagicMock(),
        "diffusers.utils": MagicMock(),
        "transformers": MagicMock(),
        "facexlib": MagicMock(),
        "basicsr": MagicMock(),
        "realesrgan": MagicMock(),
        "torch_directml": MagicMock(),
        "torch.hub": MagicMock(),
        "torchvision": MagicMock(),
        "torchvision.transforms": MagicMock(),
        "torchvision.transforms.functional": MagicMock(),
        "cv2": MagicMock(),
        "numpy": MagicMock(),
    }):
        # Setup specific mocks for imports
        sys.modules["diffusers"].StableDiffusionPipeline = MagicMock()
        sys.modules["diffusers"].StableDiffusionImg2ImgPipeline = MagicMock()
        sys.modules["diffusers"].StableDiffusionInpaintPipeline = MagicMock()
        sys.modules["diffusers"].DPMSolverMultistepScheduler = MagicMock()
        

        # Force import to ensure it's loaded and patchable
        import core.image_editor
        
        # Manually replace dependencies to guarantee they are used
        mock_pipeline_manager = MagicMock()
        mock_face_restorer = MagicMock()
        mock_memory_manager = MagicMock()
        
        # Save originals
        orig_pipeline = core.image_editor.pipeline_manager
        orig_face = core.image_editor.face_restorer
        orig_memory = core.image_editor.memory_manager
        
        try:
            core.image_editor.pipeline_manager = mock_pipeline_manager
            core.image_editor.face_restorer = mock_face_restorer
            core.image_editor.memory_manager = mock_memory_manager
            
            # Setup mocks
            mock_pipe = MagicMock()
            mock_pipeline_manager.get_img2img_pipeline.return_value = mock_pipe
            
            # Mock generation result
            mock_generated_image = Image.new("RGB", (64, 64))
            mock_pipe.return_value.images = [mock_generated_image]
            
            # Mock face restore return
            mock_face_restorer.restore_faces.return_value = mock_generated_image
            
            # Mock memory check
            mock_memory_manager.check_memory_available.return_value = (True, "OK")
            
            # Import class under test
            from core.image_editor import image_editor
            
            # Create input image
            input_image = Image.new("RGB", (64, 64))
            
            # Run method
            result, info = image_editor.edit_image(
                image=input_image,
                prompt="test prompt",
                num_steps=1,
                save_output=False
            )
            
            # Assertions
            assert result == mock_generated_image
            assert info["prompt"] == "test prompt"
            mock_pipeline_manager.get_img2img_pipeline.assert_called_once()
            mock_pipe.assert_called_once()
            
        finally:
            # Restore
            core.image_editor.pipeline_manager = orig_pipeline
            core.image_editor.face_restorer = orig_face
            core.image_editor.memory_manager = orig_memory
