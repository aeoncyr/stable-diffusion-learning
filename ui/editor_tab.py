"""
Image Editor Tab UI
Gradio interface for image-to-image editing
"""

import gradio as gr
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import generation_config, model_config
from core.image_editor import image_editor
from core.pipeline import pipeline_manager


def edit_image_handler(
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    strength: float,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    restore_faces: bool,
    progress=gr.Progress()
) -> Tuple[Image.Image, str]:
    """Handle image editing request"""
    
    if image is None:
        raise gr.Error("Please upload an image first!")
    
    if not prompt.strip():
        raise gr.Error("Please enter a prompt!")
    
    progress(0.1, desc="Loading model...")
    
    try:
        progress(0.3, desc="Processing image...")
        
        result, info = image_editor.edit_image(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            strength=strength,
            num_steps=int(num_steps),
            guidance_scale=guidance_scale,
            seed=int(seed) if seed != -1 else -1,
            restore_faces=restore_faces,
        )
        
        progress(1.0, desc="Complete!")
        
        # Format info for display
        info_text = f"""✓ Generation Complete!
        
Seed: {info['seed']}
Strength: {info['strength']}
Steps: {info['num_steps']}
Guidance: {info['guidance_scale']}
Face Restored: {info['face_restored']}
LoRA: {info['lora'] or 'None'}
Saved to: {info.get('output_path', 'Not saved')}"""
        
        return result, info_text
        
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")


def create_editor_tab() -> gr.Tab:
    """
    Create the image editor tab
    
    === EDUCATIONAL NOTE: Gradio Layouts ===
    We use `gr.Tab` to create a new tab in the UI.
    Inside, `gr.Row` creates horizontal layouts and `gr.Column` creates vertical ones.
    """
    
    with gr.Tab("🎨 Image Editor", id="editor_tab") as tab:
        gr.Markdown("""
        ## Image-to-Image Editing
        Upload an image and describe how you want to modify it. The AI will transform your image based on the prompt.
        """)
        
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    height=400,
                )
                
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the changes you want to make...",
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
                strength = gr.Slider(
                    label="Strength",
                    info="How much to change the original (0=none, 1=complete)",
                    minimum=0.1,
                    maximum=1.0,
                    value=generation_config.strength,
                    step=0.05,
                )
                
                num_steps = gr.Slider(
                    label="Steps",
                    info="More steps = better quality, slower",
                    minimum=10,
                    maximum=100,
                    value=generation_config.num_inference_steps,
                    step=1,
                )
            
            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    info="How closely to follow the prompt",
                    minimum=1.0,
                    maximum=20.0,
                    value=generation_config.guidance_scale,
                    step=0.5,
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
            fn=edit_image_handler,
            inputs=[
                input_image,
                prompt,
                negative_prompt,
                strength,
                num_steps,
                guidance_scale,
                seed,
                restore_faces,
            ],
            outputs=[output_image, info_output],
        )
    
    return tab
