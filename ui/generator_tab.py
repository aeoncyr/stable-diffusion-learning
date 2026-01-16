"""
Text-to-Image Generator Tab UI
Gradio interface for generating images from prompts
"""

import gradio as gr
from PIL import Image
from pathlib import Path
from typing import Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import generation_config, model_config
from core.generator import image_generator


def generate_handler(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    restore_faces: bool,
    progress=gr.Progress()
) -> Tuple[Image.Image, str]:
    """Handle image generation request"""
    
    if not prompt.strip():
        raise gr.Error("Please enter a prompt!")
    
    progress(0.1, desc="Loading model...")
    
    try:
        progress(0.3, desc="Generating image...")
        
        result, info = image_generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            width=int(width),
            height=int(height),
            num_steps=int(num_steps),
            guidance_scale=guidance_scale,
            seed=int(seed) if seed != -1 else -1,
            restore_faces=restore_faces,
        )
        
        progress(1.0, desc="Complete!")
        
        # Format info for display
        info_text = f"""✓ Generation Complete!
        
Seed: {info['seed']}
Size: {info['width']}x{info['height']}
Steps: {info['num_steps']}
Guidance: {info['guidance_scale']}
Face Restored: {info['face_restored']}
LoRA: {info['lora'] or 'None'}
Saved to: {info.get('output_path', 'Not saved')}"""
        
        return result, info_text
        
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")


def create_generator_tab():
    """
    Create the Text-to-Image Generator tab
    
    === EDUCATIONAL NOTE: UI Events ===
    In Gradio, we connect buttons to functions using `.click()`.
    We define `inputs` (UI components providing data) and `outputs` (UI components to update).
    """
    with gr.Tab("✨ Generator", id="generator_tab"):
        gr.Markdown("""
        ## Text-to-Image Generation
        Create images from text descriptions. Be descriptive for best results!
        """)
        
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to create...",
                    lines=4,
                )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Features to avoid (optional)...",
                    value=generation_config.default_negative_prompt,
                    lines=2,
                )
                
                with gr.Row():
                    width = gr.Dropdown(
                        label="Width",
                        choices=[256, 384, 448, 512, 576, 640, 704, 768],
                        value=generation_config.width,
                    )
                    
                    height = gr.Dropdown(
                        label="Height",
                        choices=[256, 384, 448, 512, 576, 640, 704, 768],
                        value=generation_config.height,
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
                    label="Generated Image",
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
                seed = gr.Number(
                    label="Seed",
                    info="-1 for random",
                    value=-1,
                    precision=0,
                )
                
                restore_faces = gr.Checkbox(
                    label="Restore Faces",
                    info="Apply face restoration (CodeFormer/GFPGAN)",
                    value=model_config.enable_face_restore,
                )
        
        # Connect handler
        generate_btn.click(
            fn=generate_handler,
            inputs=[
                prompt,
                negative_prompt,
                width,
                height,
                num_steps,
                guidance_scale,
                seed,
                restore_faces,
            ],
            outputs=[output_image, info_output],
        )
    
    return tab
