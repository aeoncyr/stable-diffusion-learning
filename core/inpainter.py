"""
Inpainting Module
Edit specific areas of images using masks
"""

import random
import torch
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from typing import Optional, Union, Tuple, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import generation_config, model_config, memory_config, OUTPUTS_DIR
from core.pipeline import pipeline_manager
from core.face_restore import face_restorer
from core.memory_manager import memory_manager, pre_generation_cleanup


class Inpainter:
    """
    Inpainting module using Stable Diffusion
    
    === EDUCATIONAL NOTE: Inpainting ===
    Inpainting is the process of reconstructing missing or damaged parts of an image.
    We use a 'mask' (a black and white image) to tell the AI which parts to keep (black)
    and which parts to regenerate (white).
    """
    
    def __init__(self):
        self.output_dir = OUTPUTS_DIR / "inpainted"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = None,
        num_steps: int = None,
        guidance_scale: float = None,
        seed: int = None,
        restore_faces: bool = None,
        blur_mask: int = 4,
        save_output: bool = True
    ) -> Tuple[Image.Image, dict]:
        """
        Inpaint an image based on a mask and prompt
        
        Args:
            image: Input PIL Image
            mask: Mask image (white = area to inpaint, black = preserve)
            prompt: Description of what to generate in masked area
            negative_prompt: Features to avoid
            num_steps: Number of inference steps
            guidance_scale: How closely to follow prompt
            seed: Random seed for reproducibility
            restore_faces: Apply face restoration
            blur_mask: Mask blur radius for smoother edges
            save_output: Save result to disk
        
        Returns:
            Tuple of (inpainted image, generation info dict)
        """
        # Pre-generation memory cleanup
        pre_generation_cleanup()
        
        # Apply defaults
        if negative_prompt is None:
            negative_prompt = generation_config.default_negative_prompt
        if num_steps is None:
            num_steps = generation_config.num_inference_steps
        if guidance_scale is None:
            guidance_scale = generation_config.guidance_scale
        if seed is None or seed == -1:
            seed = random.randint(0, 2147483647)
        if restore_faces is None:
            restore_faces = model_config.enable_face_restore
        
        # Get pipeline
        pipe = pipeline_manager.get_inpaint_pipeline()
        
        # Prepare image and mask
        image, mask = self._prepare_inputs(image, mask, blur_mask)
        
        # Set seed for reproducibility
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        print(f"Inpainting with prompt: '{prompt[:50]}...'")
        print(f"  Steps: {num_steps}, Guidance: {guidance_scale}")
        
        # Generate with OOM handling
        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
        except (RuntimeError, MemoryError) as e:
            error_msg = str(e).lower()
            # Check for OOM errors including DirectML-specific messages
            is_oom = any(phrase in error_msg for phrase in [
                'memory',
                'oom',
                'allocate',
                'dml allocator',
                'gpu will not respond',
                'invalid commands',
                'out of memory'
            ])
            
            if is_oom:
                memory_manager.cleanup_memory(aggressive=True)
                # Clear pipeline cache to reset device state
                pipeline_manager.clear_cache()
                
                img_size = f"{image.size[0]}x{image.size[1]}" if hasattr(image, 'size') else 'unknown'
                raise RuntimeError(
                    f"Out of Memory (DirectML) during inpainting.\n\n"
                    f"Current settings: {img_size}, {num_steps} steps\n\n"
                    f"The GPU ran out of memory. Try these solutions:\n"
                    f"1. Use a smaller input image (resize to 384x384 or smaller)\n"
                    f"2. Reduce inference steps to 15-20\n"
                    f"3. Enable 'CPU Offload' in Settings\n"
                    f"4. Close other GPU-intensive applications\n"
                    f"5. Restart the application to reset GPU state\n\n"
                    f"Original error: {e}"
                )
            raise RuntimeError(f"Inpainting failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Inpainting failed: {e}")
        
        # Apply face restoration if enabled
        if restore_faces:
            print("  Applying face restoration...")
            result = face_restorer.restore_faces(result)
        
        # Generation info
        info = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "face_restored": restore_faces,
            "mask_blur": blur_mask,
            "lora": pipeline_manager.current_lora,
        }
        
        # Save output
        if save_output:
            output_path = self._save_image(result, info)
            info["output_path"] = str(output_path)
        
        print("✓ Inpainting complete")
        return result, info
    
    def _prepare_inputs(
        self,
        image: Image.Image,
        mask: Image.Image,
        blur_radius: int
    ) -> Tuple[Image.Image, Image.Image]:
        """Prepare image and mask for inpainting with safe resolution for low VRAM"""
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert mask to grayscale
        if mask.mode != "L":
            mask = mask.convert("L")
        
        # Get original size
        w, h = image.size
        
        # Only apply VRAM-safe resizing for GPU modes (not CPU)
        # CPU mode has no VRAM limits, so allow full resolution
        is_cpu_mode = device_config.device_type == "cpu"
        max_dim = memory_config.fallback_width  # 256 for GPU, unlimited for CPU
        
        # If using GPU and image is larger than safe resolution, scale it down
        if not is_cpu_mode and (w > max_dim or h > max_dim):
            aspect = w / h
            if w > h:
                new_w = max_dim
                new_h = int(max_dim / aspect)
            else:
                new_h = max_dim
                new_w = int(max_dim * aspect)
            
            # Round to multiple of 8
            new_w = (new_w // 8) * 8
            new_h = (new_h // 8) * 8
            
            # Ensure minimum size
            new_w = max(new_w, 256)
            new_h = max(new_h, 256)
            
            print(f"  ⚠ Resizing for VRAM safety: {w}x{h} → {new_w}x{new_h}")
        else:
            # Just ensure multiple of 8
            new_w = (w // 8) * 8
            new_h = (h // 8) * 8
        
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        mask = mask.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Apply blur to mask for smooth transitions
        if blur_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
        
        return image, mask
    
    def create_mask_from_brush(
        self,
        brush_data: Union[dict, Any],
        image_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Create mask from Gradio brush/sketch data
        
        Args:
            brush_data: Data from Gradio ImageEditor component
            image_size: (width, height) of the target image
        
        Returns:
            Mask image (white = inpaint area)
        """
        # Handle different Gradio data formats
        if isinstance(brush_data, dict):
            if "mask" in brush_data:
                # Direct mask data
                mask = brush_data["mask"]
                if isinstance(mask, np.ndarray):
                    mask = Image.fromarray(mask)
                return mask.convert("L").resize(image_size)
            
            if "layers" in brush_data and brush_data["layers"]:
                # Layer-based mask
                mask = brush_data["layers"][0]
                if isinstance(mask, np.ndarray):
                    mask = Image.fromarray(mask)
                # Extract alpha channel as mask
                if mask.mode == "RGBA":
                    _, _, _, alpha = mask.split()
                    return alpha.resize(image_size)
                return mask.convert("L").resize(image_size)
        
        # Fallback: create empty mask
        return Image.new("L", image_size, 0)
    
    def _save_image(self, image: Image.Image, info: dict) -> Path:
        """Save image with metadata"""
        import time
        
        timestamp = int(time.time())
        filename = f"inpaint_{timestamp}_{info['seed']}.png"
        output_path = self.output_dir / filename
        
        # Save image
        image.save(output_path)
        
        # Save info as text file
        info_path = output_path.with_suffix(".txt")
        with open(info_path, "w") as f:
            for key, value in info.items():
                f.write(f"{key}: {value}\n")
        
        return output_path


# Singleton instance
inpainter = Inpainter()
