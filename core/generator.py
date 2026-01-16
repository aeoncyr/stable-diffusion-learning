"""
Text-to-Image Generator Module
Generate images from text prompts
"""

import random
import torch
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import generation_config, model_config, memory_config, OUTPUTS_DIR
from core.pipeline import pipeline_manager
from core.face_restore import face_restorer
from core.memory_manager import oom_protected, memory_manager, pre_generation_cleanup


class ImageGenerator:
    """
    Text-to-Image generation module
    
    === EDUCATIONAL NOTE: Text-to-Image ===
    This is the core functionality of Stable Diffusion.
    It starts with random 'noise' (static) and iteratively removes the noise
    guided by your text prompt until a clear image emerges.
    """
    
    def __init__(self):
        self.output_dir = OUTPUTS_DIR / "generated"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = None,
        width: int = None,
        height: int = None,
        num_steps: int = None,
        guidance_scale: float = None,
        seed: int = None,
        restore_faces: bool = None,
        save_output: bool = True
    ) -> Tuple[Image.Image, dict]:
        """
        Generate an image from a text prompt
        
        Args:
            prompt: Description of desired image
            negative_prompt: Features to avoid
            width: Image width (multiple of 8)
            height: Image height (multiple of 8)
            num_steps: Number of inference steps
            guidance_scale: How closely to follow prompt
            seed: Random seed for reproducibility
            restore_faces: Apply face restoration
            save_output: Save result to disk
        
        Returns:
            Tuple of (generated image, generation info dict)
        """
        # Pre-generation memory cleanup
        pre_generation_cleanup()
        
        # Apply defaults
        if negative_prompt is None:
            negative_prompt = generation_config.default_negative_prompt
        if width is None:
            width = generation_config.width
        if height is None:
            height = generation_config.height
        if num_steps is None:
            num_steps = generation_config.num_inference_steps
        if guidance_scale is None:
            guidance_scale = generation_config.guidance_scale
        if seed is None or seed == -1:
            seed = random.randint(0, 2147483647)
        if restore_faces is None:
            restore_faces = model_config.enable_face_restore
        
        # Ensure dimensions are multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        # Get pipeline
        pipe = pipeline_manager.get_txt2img_pipeline()
        
        # Set seed for reproducibility
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        print(f"Generating image with prompt: '{prompt[:50]}...'")
        print(f"  Size: {width}x{height}, Steps: {num_steps}, Guidance: {guidance_scale}")
        
        # Generate with OOM handling
        # === EDUCATIONAL NOTE: The Denoising Loop ===
        # The `pipe(...)` call triggers the main diffusion loop:
        # 1. Start with Gaussian random noise $x_T \sim \mathcal{N}(0, I)$.
        # 2. For each timestep $t$ from $T$ down to 1 (e.g., 50 steps):
        #    a. The U-Net predicts the noise $\epsilon_\theta(x_t, t, \text{cond})$ present in the current image.
        #    b. The Scheduler subtracts a portion of this noise to get $x_{t-1}$.
        #       Formula (roughly): $x_{t-1} = x_t - \text{noise\_pred} \times dt$
        # 3. After the final step, the Latent Decoder converts $x_0$ to pixel space.
        
        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
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
                # Clean up and provide helpful message
                memory_manager.cleanup_memory(aggressive=True)
                # Clear pipeline cache to reset device state
                pipeline_manager.clear_cache()
                
                raise RuntimeError(
                    f"Out of Memory (DirectML) during generation.\n\n"
                    f"Current settings: {width}x{height}, {num_steps} steps\n\n"
                    f"The GPU ran out of memory. Try these solutions:\n"
                    f"1. Reduce image size to 384x384 or smaller\n"
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
            "width": width,
            "height": height,
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
        
        print("✓ Image generation complete")
        return result, info
    
    def _save_image(self, image: Image.Image, info: dict) -> Path:
        """Save image with metadata"""
        import time
        
        timestamp = int(time.time())
        filename = f"gen_{timestamp}_{info['seed']}.png"
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
image_generator = ImageGenerator()
