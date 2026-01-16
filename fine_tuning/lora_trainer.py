"""
LoRA Trainer Module
Fine-tune Stable Diffusion with LoRA for custom subjects/styles
"""

import os
import gc
import torch
from pathlib import Path
from typing import Optional, Callable
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import training_config, model_config, MODELS_DIR
from typing import List, Tuple, Any


class LoRATrainer:
    """
    LoRA (Low-Rank Adaptation) trainer for Stable Diffusion
    Memory-efficient fine-tuning for custom subjects and styles
    
    === LoRA Mathematics ===
    Standard fine-tuning updates all weights $W$ in the model (billions of parameters).
    LoRA freezes $W$ and injects trainable rank decomposition matrices $A$ and $B$.
    
    Equation: $W' = W + \Delta W = W + BA$
    - $W \in \mathbb{R}^{d \times d}$: The frozen pre-trained weights.
    - $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$: Trainable matrices.
    - $r$: The 'rank' (e.g., 4, 8, 16). Much smaller than $d$.
    
    Result: We train <1% of parameters but achieve similar results.
    """
    
    def __init__(self):
        self.output_dir = training_config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train(
        self,
        dataset_path: Path,
        instance_prompt: str,
        learning_rate: float = None,
        max_train_steps: int = None,
        lora_rank: int = None,
        resolution: int = None,
        output_name: str = "lora_model",
        progress_callback: Optional[Callable] = None,
    ) -> Path:
        """
        Train a LoRA model on a dataset
        
        Args:
            dataset_path: Path to folder containing training images
            instance_prompt: Prompt describing the subject (e.g., "a photo of sks person")
            learning_rate: Learning rate for training
            max_train_steps: Maximum training steps
            lora_rank: Rank of LoRA matrices
            resolution: Training image resolution
            output_name: Name for output LoRA file
            progress_callback: Callback(step, total_steps, loss) for progress updates
        
        Returns:
            Path to saved LoRA model
        """
        # Apply defaults
        if learning_rate is None:
            learning_rate = training_config.learning_rate
        if max_train_steps is None:
            max_train_steps = training_config.max_train_steps
        if lora_rank is None:
            lora_rank = training_config.lora_rank
        if resolution is None:
            resolution = training_config.resolution
        
        print(f"Starting LoRA training...")
        print(f"  Dataset: {dataset_path}")
        print(f"  Prompt: {instance_prompt}")
        print(f"  Steps: {max_train_steps}")
        print(f"  LoRA Rank: {lora_rank}")
        
        try:
            # Check for required libraries
            from diffusers import StableDiffusionPipeline, DDPMScheduler
            from diffusers.loaders import LoraLoaderMixin
            from peft import LoraConfig, get_peft_model
            from transformers import CLIPTextModel, CLIPTokenizer
            
            # Load training images
            images, captions = self._load_dataset(dataset_path, instance_prompt, resolution)
            
            if len(images) == 0:
                raise ValueError(f"No images found in {dataset_path}")
            
            print(f"  Loaded {len(images)} training images")
            
            # Setup device
            device = self._get_device()
            print(f"  Device: {device}")
            
            # Load base model components
            print("  Loading model components...")
            
            pipe = StableDiffusionPipeline.from_pretrained(
                model_config.img2img_model,
                torch_dtype=torch.float32,  # Need full precision for training
                cache_dir=str(model_config.cache_dir),
                safety_checker=None,
            )
            
            # Extract components
            unet = pipe.unet
            text_encoder = pipe.text_encoder
            vae = pipe.vae
            tokenizer = pipe.tokenizer
            noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=training_config.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            
            # Apply LoRA to UNet
            unet = get_peft_model(unet, lora_config)
            unet.print_trainable_parameters()
            
            # Move to device
            unet.to(device)
            text_encoder.to(device)
            vae.to(device)
            
            # Freeze non-LoRA parameters
            vae.requires_grad_(False)
            text_encoder.requires_grad_(False)
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(
                unet.parameters(),
                lr=learning_rate,
            )
            
            # Training loop
            print("  Starting training loop...")
            unet.train()
            
            for step in range(max_train_steps):
                # Get random image
                idx = step % len(images)
                image = images[idx]
                caption = captions[idx]
                
                # Encode image
                with torch.no_grad():
                    latents = vae.encode(
                        image.unsqueeze(0).to(device)
                    ).latent_dist.sample() * 0.18215
                
                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (1,), device=device
                ).long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Encode text
                text_input = tokenizer(
                    caption,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                
                with torch.no_grad():
                    text_embeddings = text_encoder(
                        text_input.input_ids.to(device)
                    )[0]
                
                # Predict noise
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                ).sample
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Progress callback
                if progress_callback and step % 10 == 0:
                    progress_callback(step, max_train_steps, loss.item())
                
                if step % 100 == 0:
                    print(f"    Step {step}/{max_train_steps}, Loss: {loss.item():.4f}")
            
            # Save LoRA
            output_path = self.output_dir / f"{output_name}.safetensors"
            
            # Get LoRA state dict
            lora_state_dict = {}
            for name, param in unet.named_parameters():
                if "lora" in name.lower():
                    lora_state_dict[name] = param.cpu()
            
            # Save using safetensors
            from safetensors.torch import save_file
            save_file(lora_state_dict, str(output_path))
            
            print(f"✓ LoRA saved to: {output_path}")
            
            # Cleanup
            del unet, text_encoder, vae, pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return output_path
            
        except ImportError as e:
            raise ImportError(
                f"Missing required package: {e}. "
                "Please install: pip install peft bitsandbytes"
            )
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")
    
    def _load_dataset(
        self,
        dataset_path: Path,
        default_caption: str,
        resolution: int
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Load images and captions from dataset folder"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        images = []
        captions = []
        
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_path in dataset_path.glob(ext):
                try:
                    # Load image
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform(img)
                    images.append(img_tensor)
                    
                    # Load caption if exists
                    caption_path = img_path.with_suffix(".txt")
                    if caption_path.exists():
                        caption = caption_path.read_text().strip()
                    else:
                        caption = default_caption
                    captions.append(caption)
                    
                except Exception as e:
                    print(f"  Warning: Failed to load {img_path}: {e}")
        
        return images, captions
    
    def _get_device(self) -> torch.device:
        """Get training device"""
        # Try DirectML
        try:
            import torch_directml
            return torch_directml.device(0)
        except ImportError:
            pass
        
        # Try CUDA
        if torch.cuda.is_available():
            return torch.device("cuda")
        
        # Fallback to CPU
        print("  Warning: Training on CPU will be very slow!")
        return torch.device("cpu")


# Singleton instance
lora_trainer = LoRATrainer()
