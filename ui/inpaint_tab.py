"""
Inpainting Tab UI
Gradio interface for masked inpainting
"""

import gradio as gr
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import generation_config, model_config
from core.inpainter import inpainter


def inpaint_handler(
    editor_value: dict,
    prompt: str,
    negative_prompt: str,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    restore_faces: bool,
    mask_blur: int,
    progress=gr.Progress()
) -> Tuple[Image.Image, str]:
    """Handle inpainting request"""
    
    if editor_value is None:
        raise gr.Error("Please upload an image first!")
    
    if not prompt.strip():
        raise gr.Error("Please enter a prompt!")
    
    progress(0.1, desc="Processing inputs...")
    
    try:
        # Extract image and mask from editor
        # Gradio ImageEditor returns dict with 'background' and 'layers'
        if isinstance(editor_value, dict):
            if "background" in editor_value:
                image = editor_value["background"]
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
            elif "image" in editor_value:
                image = editor_value["image"]
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
            else:
                raise gr.Error("Could not extract image from editor!")
            
            # Get mask from layers or composite
            mask = None
            if "layers" in editor_value and editor_value["layers"]:
                layer = editor_value["layers"][0]
                if isinstance(layer, np.ndarray):
                    layer = Image.fromarray(layer)
                # Use alpha channel as mask
                if layer.mode == "RGBA":
                    mask = layer.split()[3]  # Alpha channel
                else:
                    mask = layer.convert("L")
            elif "composite" in editor_value:
                # Try to extract mask from composite
                composite = editor_value["composite"]
                if isinstance(composite, np.ndarray):
                    composite = Image.fromarray(composite)
                if composite.mode == "RGBA":
                    mask = composite.split()[3]
            
            if mask is None:
                raise gr.Error("Please draw a mask on the image! (Use the brush tool to mark areas to edit)")
        else:
            raise gr.Error("Invalid image format!")
        
        # Convert mask - ensure white is area to inpaint
        mask_array = np.array(mask)
        if mask_array.max() < 10:
            raise gr.Error("No mask detected! Paint on the image to mark areas to edit.")
        
        progress(0.3, desc="Loading model...")
        progress(0.5, desc="Inpainting...")
        
        result, info = inpainter.inpaint(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            num_steps=int(num_steps),
            guidance_scale=guidance_scale,
            seed=int(seed) if seed != -1 else -1,
            restore_faces=restore_faces,
            blur_mask=int(mask_blur),
        )
        
        progress(1.0, desc="Complete!")
        
        # Format info for display
        info_text = f"""✓ Inpainting Complete!
        
Seed: {info['seed']}
Steps: {info['num_steps']}
Guidance: {info['guidance_scale']}
Mask Blur: {info['mask_blur']}
Face Restored: {info['face_restored']}
LoRA: {info['lora'] or 'None'}
Saved to: {info.get('output_path', 'Not saved')}"""
        
        return result, info_text
        
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Inpainting failed: {str(e)}")


def create_inpaint_tab() -> gr.Tab:
    """
    Create the Inpainting tab
    
    === EDUCATIONAL NOTE: ImageEditor Component ===
    Gradio's `gr.ImageEditor` is powerful. It allows users to upload an image
    and draw a mask directly on it. This data is sent to our backend as a dictionary
    containing the original image and the mask layer.
    """
    with gr.Tab("🖌️ Inpainting", id="inpaint_tab") as tab:
        gr.Markdown("""
        ## Inpainting - Edit Specific Areas
        Upload an image, **use the brush to paint over areas you want to change**, then describe what should appear there.
        
        **How to use:**
        1. Upload an image
        2. Use the brush tool to paint over the area you want to edit (white = edit, black = keep)
        3. Enter a prompt describing what should appear in the painted area
        4. Click Generate!
        """)
        
        with gr.Row():
            # Left column - Editor
            with gr.Column(scale=1):
                image_editor = gr.ImageEditor(
                    label="Draw on image to mark areas to edit",
                    type="pil",
                    height=450,
                    brush=gr.Brush(
                        default_size=30,
                        colors=["#FFFFFF"],
                        default_color="#FFFFFF",
                    ),
                    eraser=gr.Eraser(default_size=30),
                    layers=True,
                )
                
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe what should appear in the masked area...",
                    lines=3,
                )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Features to avoid (optional)...",
                    value=generation_config.default_negative_prompt,
                    lines=2,
                )
                
                with gr.Row():
                    generate_btn = gr.Button(
                        "🚀 Generate",
                        variant="primary",
                        size="lg",
                    )
            
            # Right column - Output
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Output Image",
                    type="pil",
                    height=400,
                    interactive=False,
                )
                
                info_output = gr.Textbox(
                    label="Generation Info",
                    lines=8,
                    interactive=False,
                )
        
        # Advanced settings
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                num_steps = gr.Slider(
                    label="Steps",
                    info="More steps = better quality, slower",
                    minimum=10,
                    maximum=100,
                    value=generation_config.num_inference_steps,
                    step=1,
                )
                
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    info="How closely to follow the prompt",
                    minimum=1.0,
                    maximum=20.0,
                    value=generation_config.guidance_scale,
                    step=0.5,
                )
            
            with gr.Row():
                mask_blur = gr.Slider(
                    label="Mask Blur",
                    info="Blur mask edges for smoother transitions",
                    minimum=0,
                    maximum=20,
                    value=4,
                    step=1,
                )
                
                seed = gr.Number(
                    label="Seed",
                    info="-1 for random",
                    value=-1,
                    precision=0,
                )
            
            with gr.Row():
                restore_faces = gr.Checkbox(
                    label="Restore Faces",
                    info="Apply face restoration (CodeFormer/GFPGAN)",
                    value=model_config.enable_face_restore,
                )
        
        # Connect handler
        generate_btn.click(
            fn=inpaint_handler,
            inputs=[
                image_editor,
                prompt,
                negative_prompt,
                num_steps,
                guidance_scale,
                seed,
                restore_faces,
                mask_blur,
            ],
            outputs=[output_image, info_output],
        )
    
    return tab
