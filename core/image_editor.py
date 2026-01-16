"""
Image Editor Module
Image-to-image editing with Stable Diffusion
"""

import random
import torch
from PIL import Image
from pathlib import Path
from typing import Optional, Union, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import generation_config, model_config, memory_config, device_config, OUTPUTS_DIR
from core.pipeline import pipeline_manager
from core.face_restore import face_restorer
from core.memory_manager import memory_manager, pre_generation_cleanup


class ImageEditor:
    """
    Image-to-image editing using Stable Diffusion
    Supports prompt-based image modification with face preservation
    """
    
    def __init__(self):
        self.output_dir = OUTPUTS_DIR / "edited"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def edit_image(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = None,
        strength: float = None,
        num_steps: int = None,
        guidance_scale: float = None,
        seed: int = None,
        restore_faces: bool = None,
        save_output: bool = True
    ) -> Tuple[Image.Image, dict]:
        """
        Edit an image based on a text prompt
        
        Args:
            image: Input PIL Image
            prompt: Description of desired changes
            negative_prompt: Features to avoid
            strength: How much to change (0.0-1.0)
            num_steps: Number of inference steps
            guidance_scale: How closely to follow prompt
            seed: Random seed for reproducibility
            restore_faces: Apply face restoration
            save_output: Save result to disk
        
        Returns:
            Tuple of (edited image, generation info dict)
        """
        # === CONCEPT: Latent Space ===
        # Stable Diffusion doesn't work on pixels directly. It works in 'latent space'.
        # For img2img, we encode your input image into latents, add some noise (controlled by 'strength'),
        # and then let the AI 'denoise' it back into a new image based on your prompt.
        
        
        # === EDUCATIONAL NOTE: Image-to-Image Noise Injection ===
        # Unlike "Text-to-Image" which starts from pure noise ($x_T \approx 100\%$ noise),
        # "Image-to-Image" starts from the user's input image.
        # 
        # Process:
        # 1. Encoder: $z_0 = \text{VAE}(image)$
        # 2. Noise Addition: $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$
        #    - We don't add 100% noise! We add noise based on 'strength'.
        #    - If strength = 0.75, we start at step $t = 0.75 \times 50 \approx 37$.
        # 3. Denoising: The model runs for the remaining steps (37 down to 0).
        #    This allows it to keep the *structure* of the input but change the *content*.

        
        # Pre-generation memory cleanup
        pre_generation_cleanup()
        
        # Apply defaults
        if negative_prompt is None:
            negative_prompt = generation_config.default_negative_prompt
        if strength is None:
            strength = generation_config.strength
        if num_steps is None:
            num_steps = generation_config.num_inference_steps
        if guidance_scale is None:
            guidance_scale = generation_config.guidance_scale
        if seed is None or seed == -1:
            seed = random.randint(0, 2147483647)
        if restore_faces is None:
            restore_faces = model_config.enable_face_restore
        
        # Get pipeline
        pipe = pipeline_manager.get_img2img_pipeline()
        
        # Prepare image
        image = self._prepare_image(image)
        
        # Set seed for reproducibility
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        print(f"Editing image with prompt: '{prompt[:50]}...'")
        print(f"  Strength: {strength}, Steps: {num_steps}, Guidance: {guidance_scale}")
        
        # Define progress callback that returns kwargs (required by diffusers)
        def progress_callback(pipe, step_index, timestep, callback_kwargs):
            print(f"  Step {step_index+1}/{num_steps}...", end="\r", flush=True)
            return callback_kwargs

        # Generate with OOM handling
        try:
            print("  Starting generation... (this may take a moment for VAE encoding)")
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=strength,
                generator=generator,
                callback_on_step_end=progress_callback,
            ).images[0]
            print("") # Newline after progress
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
                    f"Out of Memory (DirectML) during image editing.\n\n"
                    f"Current settings: {img_size}, strength={strength}, {num_steps} steps\n\n"
                    f"The GPU ran out of memory. Try these solutions:\n"
                    f"1. Use a smaller input image (resize to 384x384 or smaller)\n"
                    f"2. Reduce inference steps to 15-20\n"
                    f"3. Enable 'CPU Offload' in Settings\n"
                    f"4. Close other GPU-intensive applications\n"
                    f"5. Restart the application to reset GPU state\n\n"
                    f"Original error: {e}"
                )
            raise RuntimeError(f"Generation failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")
        
        # Apply face restoration if enabled
        if restore_faces:
            print("  Applying face restoration...")
            result = face_restorer.restore_faces(result)
        
        # Generation info
        info = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "face_restored": restore_faces,
            "lora": pipeline_manager.current_lora,
        }
        
        # Save output
        if save_output:
            output_path = self._save_image(result, info)
            info["output_path"] = str(output_path)
        
        print("✓ Image editing complete")
        return result, info
    
    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Prepare image for processing with safe resolution for low VRAM"""
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Get original size
        w, h = image.size
        
        # Only apply VRAM-safe resizing for GPU modes (not CPU)
        # CPU mode has no VRAM limits, so allow full resolution
        is_cpu_mode = device_config.device_type == "cpu"
        
        if is_cpu_mode:
            # Even CPU has RAM limits. Cap at 1024 to prevent 6GB+ allocations
            max_dim = 1024
        else:
            max_dim = memory_config.fallback_width  # 256-512 for GPU
        
        # Resize if larger than max_dim (applies to both CPU and GPU now)
        if w > max_dim or h > max_dim:
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
            
            print(f"  ⚠ Resizing image for memory safety: {w}x{h} → {new_w}x{new_h}")
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            # Just ensure multiple of 8
            new_w = (w // 8) * 8
            new_h = (h // 8) * 8
            if (new_w, new_h) != (w, h):
                image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        return image
    
    def _save_image(self, image: Image.Image, info: dict) -> Path:
        """Save image with metadata"""
        import time
        
        timestamp = int(time.time())
        filename = f"edited_{timestamp}_{info['seed']}.png"
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
image_editor = ImageEditor()
