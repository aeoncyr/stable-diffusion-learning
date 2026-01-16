"""
Stable Diffusion Image Editor
Main application entry point

A local, unrestricted image editing application with:
- Image-to-image editing
- Inpainting
- Text-to-image generation
- LoRA fine-tuning

Optimized for AMD Radeon GPUs with DirectML support.
"""

import gradio as gr
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import ui_config, device_config, model_config, memory_config, OUTPUTS_DIR, MODELS_DIR
from ui.editor_tab import create_editor_tab
from ui.inpaint_tab import create_inpaint_tab
from ui.generator_tab import create_generator_tab
from ui.finetune_tab import create_finetune_tab
from core.pipeline import pipeline_manager
from core.memory_manager import memory_manager, get_memory_status_string


from ui.memory_monitor import create_memory_monitor

# Application CSS for better styling
CUSTOM_CSS = """
.gradio-container {
    max-width: 1400px !important;
}

.main-header {
    text-align: center;
    margin-bottom: 20px;
}

.status-bar {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 10px 20px;
    border-radius: 8px;
    margin-bottom: 20px;
}

.footer {
    text-align: center;
    margin-top: 20px;
    color: #666;
}
"""


def create_app() -> gr.Blocks:
    """
    Create the main Gradio application
    
    # === The User Interface (Gradio) ===
    # This file builds the web interface.
    # Gradio works on an "Event-Driven" architecture:
    # 1. **Components**: The building blocks (Buttons, Slicers, Image inputs).
    # 2. **Events**: Actions like `.click()` or `.change()`.
    # 3. **The Queue**: When you click 'Generate', your request goes into a queue.
    #    The backend processes requests one by one to prevent crashing the GPU.
    #
    # Visualization:
    # UI Input -> [Queue] -> [GPU Backend: Pipeline] -> [VRAM Allocation] -> [Inference Loop] -> UI Output

    # === Gradio Blocks ===
    # Gradio 'Blocks' is a low-level API that allows for custom layouts.
    # Unlike the simple 'Interface' API, Blocks lets us arrange rows, columns,
    # and tabs exactly how we want them.
    """
    
    with gr.Blocks(
        title="SD Image Editor",
        theme=gr.themes.Soft(), # Soft theme is easy on the eyes
        css=CUSTOM_CSS,
    ) as app:
        
        # Header
        gr.Markdown(
            """
            # 🎨 Stable Diffusion Image Editor
            
            **Unrestricted local image editing with AI** | Edit images, inpaint areas, generate from text, and train custom LoRA models.
            """,
            elem_classes=["main-header"]
        )
        
        # Status bar with memory info
        with gr.Row(elem_classes=["status-bar"]):
            with gr.Column(scale=3):
                device_info = str(device_config.get_device())
                device_display = "AMD GPU (DirectML)" if "privateuseone" in device_info.lower() else device_info.upper()
                
                gr.Markdown(
                    f"""
                    **Device:** {device_display} | 
                    **Memory Optimizations:** {'Enabled' if device_config.enable_attention_slicing else 'Disabled'} |
                    **OOM Protection:** {'On' if memory_config.enable_oom_protection else 'Off'} |
                    **Face Restore:** {'On' if model_config.enable_face_restore else 'Off'} |
                    **LoRA:** {pipeline_manager.current_lora or 'None'}
                    """
                )
            
            with gr.Column(scale=1):
                # Dynamic memory monitor
                create_memory_monitor()
        
        # Main tabs
        with gr.Tabs():
            create_editor_tab()
            create_inpaint_tab()
            create_generator_tab()
            create_finetune_tab()
            
            # Settings tab
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("### Application Settings")
                
                # Device mode selection
                gr.Markdown("#### Device Mode")
                gr.Markdown("*If you keep getting OOM errors, switch to CPU mode. It's slow (~5-10 min per image) but won't crash.*")
                
                device_mode = gr.Radio(
                    label="Processing Device",
                    choices=["auto", "directml", "cpu"],
                    value=device_config.device_type,
                    info="CPU mode is slow but never runs out of memory"
                )
                
                gr.Markdown("---")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Memory Optimization")
                        
                        attention_slicing = gr.Checkbox(
                            label="Enable Attention Slicing",
                            value=device_config.enable_attention_slicing,
                            info="Reduces memory usage per attention layer"
                        )
                        
                        vae_tiling = gr.Checkbox(
                            label="Enable VAE Tiling",
                            value=device_config.enable_vae_tiling,
                            info="Process large images in tiles"
                        )
                        
                        vae_slicing = gr.Checkbox(
                            label="Enable VAE Slicing",
                            value=device_config.enable_vae_slicing,
                            info="Process VAE in slices for batch processing"
                        )
                        
                        cpu_offload = gr.Checkbox(
                            label="Enable CPU Offload",
                            value=device_config.enable_cpu_offload,
                            info="Offload model layers to CPU (slower but uses less VRAM)"
                        )
                        
                        use_fp16 = gr.Checkbox(
                            label="Use Half Precision (FP16)",
                            value=device_config.use_float16,
                            info="Halves memory usage with minimal quality loss"
                        )
                    
                    with gr.Column():
                        gr.Markdown("#### Advanced Optimizations")
                        
                        enable_tome = gr.Checkbox(
                            label="Enable Token Merging (ToMe)",
                            value=device_config.enable_tomesd,
                            info="Merges redundant tokens to save ~30-50% memory"
                        )
                        
                        tome_ratio = gr.Slider(
                            label="Token Merge Ratio",
                            minimum=0.1,
                            maximum=0.75,
                            value=device_config.tomesd_ratio,
                            step=0.1,
                            info="Higher = more memory savings, slightly lower quality"
                        )
                        
                        tiny_vae = gr.Checkbox(
                            label="Use Tiny VAE",
                            value=device_config.use_tiny_vae,
                            info="Smaller VAE for less memory (lower quality)"
                        )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Face Restoration")
                        
                        face_restore = gr.Checkbox(
                            label="Enable Face Restoration",
                            value=model_config.enable_face_restore,
                            info="Apply CodeFormer/GFPGAN to preserve facial features"
                        )
                        
                        face_model = gr.Radio(
                            label="Face Restore Model",
                            choices=["codeformer", "gfpgan", "both"],
                            value=model_config.face_restore_model,
                        )
                        
                        codeformer_weight = gr.Slider(
                            label="CodeFormer Fidelity",
                            minimum=0.0,
                            maximum=1.0,
                            value=model_config.codeformer_fidelity,
                            step=0.1,
                            info="0=quality, 1=fidelity to original"
                        )
                
                gr.Markdown("---")
                gr.Markdown("#### OOM (Out of Memory) Protection")
                
                with gr.Row():
                    with gr.Column():
                        oom_protection = gr.Checkbox(
                            label="Enable OOM Protection",
                            value=memory_config.enable_oom_protection,
                            info="Automatically handle memory errors with retry and reduced settings"
                        )
                        
                        auto_cleanup = gr.Checkbox(
                            label="Auto Memory Cleanup",
                            value=memory_config.auto_cleanup_enabled,
                            info="Clean up memory before each generation"
                        )
                    
                    with gr.Column():
                        fallback_res = gr.Slider(
                            label="Fallback Resolution",
                            minimum=256,
                            maximum=512,
                            value=memory_config.fallback_width,
                            step=64,
                            info="Resolution to use when OOM occurs"
                        )
                        
                        max_retries = gr.Slider(
                            label="Max OOM Retries",
                            minimum=1,
                            maximum=5,
                            value=memory_config.max_oom_retries,
                            step=1,
                            info="Number of retries with reduced settings"
                        )
                
                with gr.Row():
                    save_settings_btn = gr.Button("💾 Save Settings", variant="primary")
                    clear_cache_btn = gr.Button("🗑️ Clear Model Cache", variant="secondary")
                    refresh_mem_btn = gr.Button("🔄 Refresh Memory Status")
                
                settings_status = gr.Textbox(label="Status", interactive=False)
                
                def save_settings(
                    dev_mode,
                    att_slice, vae_tile, vae_slice, cpu_off, fp16, 
                    tome_enabled, tome_ratio_val, tiny_vae_enabled,
                    face_rest, face_mod, cf_weight,
                    oom_prot, auto_clean, fallback, retries
                ):
                    # Device mode (most important for OOM prevention)
                    device_changed = device_config.device_type != dev_mode
                    device_config.device_type = dev_mode
                    
                    # Core optimizations
                    device_config.enable_attention_slicing = att_slice
                    device_config.enable_vae_tiling = vae_tile
                    device_config.enable_vae_slicing = vae_slice
                    device_config.enable_cpu_offload = cpu_off
                    device_config.use_float16 = fp16
                    
                    # Advanced optimizations
                    device_config.enable_tomesd = tome_enabled
                    device_config.tomesd_ratio = tome_ratio_val
                    device_config.use_tiny_vae = tiny_vae_enabled
                    
                    # Face restoration
                    model_config.enable_face_restore = face_rest
                    model_config.face_restore_model = face_mod
                    model_config.codeformer_fidelity = cf_weight
                    
                    # OOM protection
                    memory_config.enable_oom_protection = oom_prot
                    memory_config.auto_cleanup_enabled = auto_clean
                    memory_config.fallback_width = int(fallback)
                    memory_config.fallback_height = int(fallback)
                    memory_config.max_oom_retries = int(retries)
                    
                    # Clear cache to apply new settings
                    pipeline_manager.clear_cache()
                    
                    # Reinitialize pipeline manager if device changed
                    if device_changed:
                        pipeline_manager._init_device()
                        status_prefix = f"⚠ Device changed to {dev_mode.upper()}. "
                    else:
                        status_prefix = "✓ "
                    
                    if dev_mode == "cpu":
                        return f"{status_prefix}CPU mode enabled. Generation will be SLOW (~5-10 min) but no OOM errors."
                    
                    optimizations = []
                    if att_slice: optimizations.append("Attention Slicing")
                    if vae_tile: optimizations.append("VAE Tiling")
                    if vae_slice: optimizations.append("VAE Slicing")
                    if tiny_vae_enabled: optimizations.append("Tiny VAE")
                    
                    opt_str = ", ".join(optimizations) if optimizations else "None"
                    return f"{status_prefix}Settings saved! Device: {dev_mode.upper()}, Active: {opt_str}"
                
                def clear_cache():
                    pipeline_manager.clear_cache()
                    memory_manager.cleanup_memory(aggressive=True)
                    return f"✓ Model cache cleared! {get_memory_status_string()}"
                
                def refresh_memory():
                    return f"📊 {get_memory_status_string()}"
                
                save_settings_btn.click(
                    fn=save_settings,
                    inputs=[
                        device_mode,
                        attention_slicing, vae_tiling, vae_slicing, cpu_offload, use_fp16,
                        enable_tome, tome_ratio, tiny_vae,
                        face_restore, face_model, codeformer_weight,
                        oom_protection, auto_cleanup, fallback_res, max_retries
                    ],
                    outputs=[settings_status]
                )
                
                clear_cache_btn.click(
                    fn=clear_cache,
                    outputs=[settings_status]
                )
                
                refresh_mem_btn.click(
                    fn=refresh_memory,
                    outputs=[settings_status]
                )
        
        # Footer
        gr.Markdown(
            f"""
            ---
            **Output Directory:** `{OUTPUTS_DIR}` | **Models Directory:** `{MODELS_DIR}`
            
            *Models are downloaded from Hugging Face on first use. No content restrictions applied.*
            """,
            elem_classes=["footer"]
        )
    
    return app


def main():
    """Run the application"""
    print("=" * 60)
    print("  Stable Diffusion Image Editor")
    print("  Optimized for AMD Radeon with DirectML")
    print("=" * 60)
    
    # Create app
    app = create_app()
    
    # Launch
    print(f"\nStarting server on http://{ui_config.server_name}:{ui_config.server_port}")
    print("Press Ctrl+C to stop\n")
    
    app.launch(
        server_name=ui_config.server_name,
        server_port=ui_config.server_port,
        share=ui_config.share,
        inbrowser=ui_config.inbrowser,
    )


if __name__ == "__main__":
    main()
