"""
Fine-Tuning Tab UI
Gradio interface for LoRA training and model management
"""

import gradio as gr
from pathlib import Path
from typing import List, Tuple, Optional
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import training_config, model_config, DATASETS_DIR
from core.pipeline import pipeline_manager


# Training state
training_in_progress = False
training_logs = []


def get_dataset_folders() -> List[str]:
    """Get available dataset folders"""
    folders = []
    if DATASETS_DIR.exists():
        for item in DATASETS_DIR.iterdir():
            if item.is_dir():
                # Count images
                images = list(item.glob("*.jpg")) + list(item.glob("*.png")) + list(item.glob("*.jpeg"))
                folders.append(f"{item.name} ({len(images)} images)")
    return folders if folders else ["No datasets found - create a folder in datasets/"]


def get_available_loras() -> List[str]:
    """Get available LoRA models"""
    loras = pipeline_manager.get_available_loras()
    return loras if loras else ["No LoRA models found"]


def refresh_lists() -> Tuple[gr.Dropdown, gr.Dropdown]:
    """Refresh dataset and LoRA lists"""
    datasets = get_dataset_folders()
    loras = get_available_loras()
    return (
        gr.Dropdown(choices=datasets, value=datasets[0] if datasets else None),
        gr.Dropdown(choices=loras, value=loras[0] if loras else None)
    )


def load_lora_handler(lora_name: str) -> str:
    """Load a LoRA model"""
    if not lora_name or lora_name == "No LoRA models found":
        return "⚠️ No LoRA selected"
    
    try:
        lora_path = model_config.lora_dir / lora_name
        pipeline_manager.load_lora(lora_path)
        return f"✓ LoRA loaded: {lora_name}\n\nThis LoRA will now be applied to all generations."
    except Exception as e:
        return f"❌ Failed to load LoRA: {str(e)}"


def unload_lora_handler() -> str:
    """Unload current LoRA"""
    try:
        pipeline_manager.unload_lora()
        return "✓ LoRA unloaded. Generations will use base model."
    except Exception as e:
        return f"❌ Failed to unload: {str(e)}"


def start_training_handler(
    dataset_folder: str,
    instance_prompt: str,
    learning_rate: float,
    train_steps: int,
    lora_rank: int,
    resolution: int,
    output_name: str,
    progress=gr.Progress()
) -> str:
    """Start LoRA training"""
    global training_in_progress, training_logs
    
    if training_in_progress:
        return "❌ Training already in progress!"
    
    if not dataset_folder or "No datasets" in dataset_folder:
        return "❌ Please select a dataset folder!"
    
    if not instance_prompt.strip():
        return "❌ Please enter an instance prompt!"
    
    if not output_name.strip():
        return "❌ Please enter an output name for the LoRA!"
    
    # Extract folder name
    folder_name = dataset_folder.split(" (")[0]
    dataset_path = DATASETS_DIR / folder_name
    
    if not dataset_path.exists():
        return f"❌ Dataset folder not found: {folder_name}"
    
    training_in_progress = True
    training_logs = []
    
    try:
        progress(0.1, desc="Preparing training...")
        
        # Import training module
        from fine_tuning.lora_trainer import LoRATrainer
        
        trainer = LoRATrainer()
        
        progress(0.2, desc="Loading model...")
        
        # Training callback
        def update_progress(step, total_steps, loss):
            pct = 0.2 + (step / total_steps) * 0.7
            progress(pct, desc=f"Step {step}/{total_steps}, Loss: {loss:.4f}")
            training_logs.append(f"Step {step}: loss={loss:.4f}")
        
        # Start training
        output_path = trainer.train(
            dataset_path=dataset_path,
            instance_prompt=instance_prompt,
            learning_rate=learning_rate,
            max_train_steps=train_steps,
            lora_rank=int(lora_rank),
            resolution=resolution,
            output_name=output_name,
            progress_callback=update_progress,
        )
        
        progress(1.0, desc="Complete!")
        
        training_in_progress = False
        return f"""✓ Training Complete!

LoRA saved to: {output_path}

Training Summary:
- Dataset: {folder_name}
- Instance Prompt: {instance_prompt}
- Steps: {train_steps}
- Learning Rate: {learning_rate}
- LoRA Rank: {lora_rank}

You can now load this LoRA in the "Load LoRA" section above."""
        
    except Exception as e:
        training_in_progress = False
        return f"❌ Training failed: {str(e)}\n\nCheck the console for more details."


def create_dataset_instructions() -> str:
    """Generate dataset creation instructions"""
    return f"""### How to Prepare Training Data

1. Create a new folder in: `{DATASETS_DIR}`
2. Add 5-20 high-quality images of your subject
3. Images should be:
   - Clear and well-lit
   - Showing the subject from different angles
   - Minimum 512x512 pixels
   - JPG or PNG format

4. Optionally, create caption files:
   - For each `image.jpg`, create `image.txt`
   - Write a description of the image

Example folder structure:
```
datasets/
└── my_character/
    ├── image1.jpg
    ├── image1.txt (optional caption)
    ├── image2.jpg
    └── image2.txt (optional caption)
```
"""


def create_finetune_tab() -> gr.Tab:
    """
    Create the LoRA Fine-Tuning tab
    
    === EDUCATIONAL NOTE: LoRA Training ===
    LoRA (Low-Rank Adaptation) is a technique to fine-tune large models like Stable Diffusion
    without retraining the entire model. It's much faster and requires less memory.
    """
    with gr.Tab("🔧 LoRA Fine-Tuning", id="finetune_tab"):
        gr.Markdown("""
        ## Fine-Tuning with LoRA
        Train custom LoRA models to capture specific styles or subjects.
        """)
        
        with gr.Tabs():
            # Model Management Tab
            with gr.Tab("📦 Load LoRA"):
                gr.Markdown("### Load a trained LoRA model to use in generations")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        lora_dropdown = gr.Dropdown(
                            label="Available LoRA Models",
                            choices=get_available_loras(),
                            interactive=True,
                        )
                    
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("🔄 Refresh", size="sm")
                
                with gr.Row():
                    load_btn = gr.Button("📥 Load LoRA", variant="primary")
                    unload_btn = gr.Button("📤 Unload LoRA", variant="secondary")
                
                lora_status = gr.Textbox(
                    label="Status",
                    lines=4,
                    interactive=False,
                    value=f"Current LoRA: {pipeline_manager.current_lora or 'None'}"
                )
                
                # Connect handlers
                load_btn.click(
                    fn=load_lora_handler,
                    inputs=[lora_dropdown],
                    outputs=[lora_status],
                )
                
                unload_btn.click(
                    fn=unload_lora_handler,
                    outputs=[lora_status],
                )
            
            # Training Tab
            with gr.Tab("🎓 Train LoRA"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Training Configuration")
                        
                        dataset_dropdown = gr.Dropdown(
                            label="Dataset Folder",
                            choices=get_dataset_folders(),
                            interactive=True,
                        )
                        
                        instance_prompt = gr.Textbox(
                            label="Instance Prompt",
                            placeholder="e.g., 'a photo of sks person' or 'artwork in xyz style'",
                            info="Use a unique token like 'sks' to identify your subject",
                        )
                        
                        output_name = gr.Textbox(
                            label="Output LoRA Name",
                            placeholder="e.g., 'my_character_lora'",
                        )
                        
                        with gr.Row():
                            learning_rate = gr.Number(
                                label="Learning Rate",
                                value=training_config.learning_rate,
                                precision=6,
                            )
                            
                            train_steps = gr.Slider(
                                label="Training Steps",
                                minimum=100,
                                maximum=5000,
                                value=training_config.max_train_steps,
                                step=100,
                            )
                        
                        with gr.Row():
                            lora_rank = gr.Slider(
                                label="LoRA Rank",
                                info="Higher = more capacity, larger file",
                                minimum=1,
                                maximum=128,
                                value=training_config.lora_rank,
                                step=1,
                            )
                            
                            resolution = gr.Dropdown(
                                label="Training Resolution",
                                choices=[256, 384, 512, 768],
                                value=training_config.resolution,
                            )
                        
                        train_btn = gr.Button(
                            "🚀 Start Training",
                            variant="primary",
                            size="lg",
                        )
                    
                    with gr.Column():
                        training_output = gr.Textbox(
                            label="Training Output",
                            lines=15,
                            interactive=False,
                        )
                
                # Connect training handler
                train_btn.click(
                    fn=start_training_handler,
                    inputs=[
                        dataset_dropdown,
                        instance_prompt,
                        learning_rate,
                        train_steps,
                        lora_rank,
                        resolution,
                        output_name,
                    ],
                    outputs=[training_output],
                )
            
            # Help Tab
            with gr.Tab("📖 Help"):
                gr.Markdown(create_dataset_instructions())
        
        # Refresh handler for both dropdowns
        def do_refresh():
            datasets = get_dataset_folders()
            loras = get_available_loras()
            return gr.Dropdown(choices=loras), gr.Dropdown(choices=datasets)
        
        refresh_btn.click(
            fn=do_refresh,
            outputs=[lora_dropdown, dataset_dropdown],
        )
    
    return tab
