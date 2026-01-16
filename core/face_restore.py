"""
Face Restoration Module
Supports CodeFormer and GFPGAN for preserving facial features
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Union, Literal

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# Compatibility patch for torchvision 0.17+ (functional_tensor removed)
# Must be applied BEFORE importing basicsr/gfpgan
# ============================================================
try:
    import torchvision.transforms.functional_tensor
except ModuleNotFoundError:
    # In torchvision >= 0.17, functional_tensor was merged into functional
    # Create a shim module to maintain backward compatibility
    import torchvision.transforms.functional as F
    import types
    
    # Create a fake module with the same functions
    functional_tensor = types.ModuleType('torchvision.transforms.functional_tensor')
    functional_tensor.rgb_to_grayscale = F.rgb_to_grayscale
    
    # Register it in sys.modules so imports find it
    sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor
    
    print("✓ Applied torchvision compatibility patch")
# ============================================================

from config import model_config, MODELS_DIR

# Import gc for memory cleanup
import gc


class FaceRestorer:
    """
    Face restoration using CodeFormer and/or GFPGAN
    Automatically detects and enhances faces in images
    """
    
    def __init__(self):
        self._gfpgan = None
        self._codeformer = None
        self._face_helper = None
        
        # Model paths
        self.model_dir = MODELS_DIR / "face_restore"
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_gfpgan(self) -> None:
        """Initialize GFPGAN model"""
        if self._gfpgan is not None:
            return
        
        try:
            from gfpgan import GFPGANer
            
            model_path = self.model_dir / "GFPGANv1.4.pth"
            
            # Download if not exists
            if not model_path.exists():
                print("Downloading GFPGAN model...")
                import urllib.request
                url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
                urllib.request.urlretrieve(url, str(model_path))
                print("✓ GFPGAN model downloaded")
            
            self._gfpgan = GFPGANer(
                model_path=str(model_path),
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            print("✓ GFPGAN initialized")
            
        except Exception as e:
            print(f"⚠ GFPGAN initialization failed: {e}")
            self._gfpgan = None
    
    def _init_codeformer(self) -> None:
        """Initialize CodeFormer model"""
        if self._codeformer is not None:
            return
        
        try:
            # CodeFormer uses basicsr and facexlib
            from basicsr.utils import imwrite
            from basicsr.utils.download_util import load_file_from_url
            
            # Try to import codeformer
            try:
                from codeformer.basicsr.utils.registry import ARCH_REGISTRY
                from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
            except ImportError:
                # Alternative import path
                try:
                    from .codeformer_arch import CodeFormer
                except ImportError:
                    pass # Will handle later
                from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            
            # Download CodeFormer model
            model_path = self.model_dir / "codeformer.pth"
            if not model_path.exists():
                print("Downloading CodeFormer model...")
                load_file_from_url(
                    url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
                    model_dir=str(self.model_dir),
                    progress=True,
                    file_name="codeformer.pth"
                )
                print("✓ CodeFormer model downloaded")
            
            import torch
            try:
                # Try local import first (downloaded files)
                from .codeformer_arch import CodeFormer
                print("✓ Loaded CodeFormer from local file")
            except ImportError:
                # Fallback to system install
                from basicsr.archs.codeformer_arch import CodeFormer
                
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            
            # Initialize model
            self._codeformer = CodeFormer(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=['32', '64', '128', '256']
            )
            
            # Load weights
            checkpoint = torch.load(str(model_path), map_location='cpu')
            self._codeformer.load_state_dict(checkpoint['params_ema'])
            self._codeformer.eval()
            
            # Move to appropriate device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._codeformer = self._codeformer.to(device)
            
            # Initialize face helper
            self._face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                device=device
            )
            
            print("✓ CodeFormer initialized")
            
        except Exception as e:
            print(f"⚠ CodeFormer initialization failed: {e}")
            print("  Falling back to GFPGAN only")
            self._codeformer = None
    
    def restore_faces(
        self,
        image: Union[Image.Image, np.ndarray],
        method: Literal["gfpgan", "codeformer", "both"] = None,
        codeformer_fidelity: float = None
    ) -> Image.Image:
        """
        Restore faces in an image
        
        Args:
            image: Input image (PIL or numpy)
            method: Restoration method (default from config)
            codeformer_fidelity: CodeFormer fidelity (0=quality, 1=fidelity)
        
        Returns:
            Image with restored faces
        """
        if method is None:
            method = model_config.face_restore_model
        
        if codeformer_fidelity is None:
            codeformer_fidelity = model_config.codeformer_fidelity
        
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            # RGB to BGR for OpenCV
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_np = image.copy()
        
        result = img_np.copy()
        
        # Apply GFPGAN
        if method in ["gfpgan", "both"]:
            result = self._restore_gfpgan(result)
        
        # Apply CodeFormer
        if method in ["codeformer", "both"]:
            result = self._restore_codeformer(result, codeformer_fidelity)
        
        # Convert back to PIL
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # === EDUCATIONAL NOTE: Memory Cleanup ===
        # Image processing creates large temporary arrays.
        # We manually trigger garbage collection to prevent memory creeping up.
        gc.collect()
        
        return Image.fromarray(result_rgb)
    
    def _restore_gfpgan(self, img: np.ndarray) -> np.ndarray:
        """Apply GFPGAN restoration"""
        self._init_gfpgan()
        
        if self._gfpgan is None:
            return img
        
        try:
            _, _, output = self._gfpgan.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
            return output
        except Exception as e:
            print(f"⚠ GFPGAN processing failed: {e}")
            return img
    
    def _restore_codeformer(
        self,
        img: np.ndarray,
        fidelity: float = 0.5
    ) -> np.ndarray:
        """Apply CodeFormer restoration"""
        self._init_codeformer()
        
        if self._codeformer is None or self._face_helper is None:
            return img
        
        try:
            import torch
            
            self._face_helper.clean_all()
            self._face_helper.read_image(img)
            
            # Detect faces
            num_faces = self._face_helper.get_face_landmarks_5(
                only_center_face=False,
                resize=640,
                eye_dist_threshold=5
            )
            
            if num_faces == 0:
                return img
            
            # Align and warp faces
            self._face_helper.align_warp_face()
            
            # Process each face
            for idx, cropped_face in enumerate(self._face_helper.cropped_faces):
                # Preprocess
                cropped_face_t = torch.from_numpy(
                    cropped_face.transpose(2, 0, 1)
                ).float().unsqueeze(0) / 255.0
                
                device = next(self._codeformer.parameters()).device
                cropped_face_t = cropped_face_t.to(device)
                
                # Inference
                with torch.no_grad():
                    output = self._codeformer(
                        cropped_face_t,
                        w=fidelity,
                        adain=True
                    )[0]
                
                # Post-process
                restored_face = output.squeeze(0).cpu().numpy()
                restored_face = np.clip(
                    restored_face.transpose(1, 2, 0) * 255,
                    0, 255
                ).astype(np.uint8)
                
                self._face_helper.add_restored_face(restored_face)
            
            # Paste back
            self._face_helper.get_inverse_affine(None)
            restored_img = self._face_helper.paste_faces_to_input_image()
            
            return restored_img
            
        except Exception as e:
            print(f"⚠ CodeFormer processing failed: {e}")
            # Clean up on error
            gc.collect()
            return img
    
    def has_faces(self, image: Union[Image.Image, np.ndarray]) -> bool:
        """Check if image contains faces"""
        try:
            from facexlib.detection import init_detection_model
            
            # Initialize detector if needed
            detector = init_detection_model('retinaface_resnet50', device='cpu')
            
            # Convert to numpy
            if isinstance(image, Image.Image):
                img_np = np.array(image)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                img_np = image
            
            # Detect faces
            bboxes = detector.detect_faces(img_np, 0.5)
            
            return len(bboxes) > 0
            
        except Exception as e:
            print(f"⚠ Face detection failed: {e}")
            # Assume there might be faces
            return True


# Singleton instance
face_restorer = FaceRestorer()
