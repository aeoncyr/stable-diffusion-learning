"""
Stable Diffusion Image Editor - Configuration
Optimized for AMD Radeon GPUs with DirectML
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal

# Base paths
BASE_DIR = Path(__file__).parent.resolve()

# Allow environment variables to override paths
MODELS_DIR = Path(os.getenv("SD_MODELS_DIR", BASE_DIR / "models"))
OUTPUTS_DIR = Path(os.getenv("SD_OUTPUTS_DIR", BASE_DIR / "outputs"))
DATASETS_DIR = Path(os.getenv("SD_DATASETS_DIR", BASE_DIR / "datasets"))

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)


@dataclass
class DeviceConfig:
    """
    Device configuration with AMD DirectML support
    
    === Dataclasses ===
    We use @dataclass to automatically generate __init__, __repr__, and other methods.
    It's cleaner than writing standard classes for simple data containers.
    """
    
    # Device options: "directml", "cuda", "cpu", "auto"
    # === DirectML ===
    # DirectML is an API from Microsoft that allows hardware acceleration on
    # any DirectX 12 compatible GPU (including AMD Radeon). 
    # This is crucial because standard CUDA only works on NVIDIA cards.
    device_type: Literal["directml", "cuda", "cpu", "auto"] = "cpu"
    
    # DirectML device index (for multi-GPU systems)
    directml_device_id: int = 0
    
    # === Core Memory Optimizations ===
    enable_attention_slicing: bool = True
    attention_slice_size: Literal["auto", "max", 1, 2, 4, 8] = 1  # 1 = minimum memory
    enable_vae_tiling: bool = True
    enable_vae_slicing: bool = True  # NEW: Process VAE in slices
    enable_cpu_offload: bool = False  # Disabled - doesn't work with DirectML
    use_float16: bool = True  # Half precision for memory savings
    
    # === Advanced Optimizations ===
    # Token Merging (ToMe) - merges redundant tokens to save memory
    # NOTE: Disabled by default - not compatible with DirectML (AMD GPUs)
    enable_tomesd: bool = False  # Only enable for CUDA/NVIDIA GPUs
    tomesd_ratio: float = 0.5  # 0.0-1.0, higher = more merging, less memory
    
    # Tiny VAE - use smaller VAE for less memory (slightly lower quality)
    use_tiny_vae: bool = False  # NEW: Use Tiny AutoEncoder for VAE
    
    # torch.compile for optimization (requires PyTorch 2.0+)
    enable_torch_compile: bool = False  # NEW: Can cause issues with DirectML
    
    def get_device(self):
        """Get the appropriate device for PyTorch"""
        if self.device_type == "auto":
            return self._auto_detect_device()
        elif self.device_type == "directml":
            try:
                import torch_directml
                return torch_directml.device(self.directml_device_id)
            except ImportError:
                print("DirectML not available, falling back to CPU")
                return "cpu"
        elif self.device_type == "cuda":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"
    
    def _auto_detect_device(self):
        """Auto-detect best available device"""
        # Try DirectML first (for AMD GPUs on Windows)
        try:
            import torch_directml
            print("✓ Using DirectML (AMD GPU)")
            return torch_directml.device(self.directml_device_id)
        except ImportError:
            pass
        
        # Try CUDA (for NVIDIA GPUs)
        try:
            import torch
            if torch.cuda.is_available():
                print("✓ Using CUDA (NVIDIA GPU)")
                return "cuda"
        except:
            pass
        
        # Fallback to CPU
        print("⚠ Using CPU (slower performance)")
        return "cpu"


@dataclass
class ModelConfig:
    """Model configuration"""
    
    # Photorealistic models for best quality
    # Realistic Vision V5.1 - highly rated for photorealism
    # Using Realistic Vision for both img2img and inpainting for consistent realism
    inpainting_model: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    img2img_model: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    
    # Scheduler settings
    scheduler_type: Literal["dpm++", "euler_a"] = "dpm++"
    
    # Local model cache
    cache_dir: Path = field(default_factory=lambda: MODELS_DIR / "hub")
    
    # Face restoration - higher fidelity for better face preservation
    enable_face_restore: bool = True
    face_restore_model: Literal["codeformer", "gfpgan", "both"] = "codeformer"
    codeformer_fidelity: float = 0.7  # 0=quality, 1=fidelity (0.7 = balanced realism)
    
    # Custom LoRA models directory
    lora_dir: Path = field(default_factory=lambda: MODELS_DIR / "lora")
    
    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lora_dir.mkdir(parents=True, exist_ok=True)


@dataclass  
class GenerationConfig:
    """Default generation parameters - optimized for quality over speed"""
    
    # Image-to-image
    """
    Default generation parameters
    """
    
    # === Noise Strength (Img2Img) ===
    # Determines the initial noise level added to the latent image.
    # Functionally: 
    # - 0.0: No change (image remains exact same)
    # - 1.0: Complete destruction (image becomes random noise, heavily altered)
    strength: float = 0.75
    
    # === Inference Steps ===
    # The number of steps the ODE solver takes to integrate from noise to image.
    # More steps = smaller $dt$ in the differential equation = lower discretization error.
    # Functionally:
    # - Low (10-20): Fast, but rough details.
    # - Med (30-50): Good balance (standard).
    # - High (50+): Diminishing returns, very slow.
    num_inference_steps: int = 50
    
    # === Guidance Scale (CFG) ===
    # Classifier-Free Guidance scale ($w$).
    # Functionally:
    # - High (>10): Forces partial adherence to prompt, can cause artifacts/frying.
    # - Low (<5): More creative/relaxed, but might miss prompt details.
    guidance_scale: float = 7.0
    
    # === Clip Skip ===
    # Skips the last N layers of the CLIP text encoder.
    # Functionally: 
    # - 1: Use final layer (accurate).
    # - 2: Use second-to-last layer (popular for anime/creative models).
    clip_skip: int = 1
    
    default_negative_prompt: str = (
        "(worst quality, low quality:1.4), blurry, bad quality, distorted, "
        "deformed, ugly, bad anatomy, bad proportions, extra limbs, cloned face, "
        "disfigured, gross proportions, malformed limbs, missing arms, missing legs, "
        "extra arms, extra legs, fused fingers, too many fingers, long neck, "
        "out of frame, watermark, signature, text, logo, cropped, "
        "cartoon, anime, illustration, painting, drawing, art, "
        "3d render, CGI, oversaturated, overexposed, underexposed"
    )
    
    # Image size (512 for SD 1.5 optimal quality)
    width: int = 512
    height: int = 512
    
    # Seed for reproducibility (-1 for random)
    seed: int = -1


@dataclass
class UIConfig:
    """UI configuration"""
    
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    share: bool = False  # Set True to create public link
    theme: str = "soft"  # Gradio theme
    
    # Auto-open browser
    inbrowser: bool = True


@dataclass
class TrainingConfig:
    """LoRA fine-tuning configuration"""
    
    # Training parameters
    learning_rate: float = 1e-4
    train_batch_size: int = 1
    max_train_steps: int = 1000
    
    # LoRA parameters
    lora_rank: int = 4  # Lower = smaller file, 4-8 recommended
    lora_alpha: int = 4
    
    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"
    
    # Image resolution
    resolution: int = 512
    
    # Output
    output_dir: Path = field(default_factory=lambda: MODELS_DIR / "lora_trained")
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class MemoryConfig:
    """Memory management and OOM prevention configuration"""
    
    # Enable OOM protection
    enable_oom_protection: bool = True
    
    # Maximum memory usage before preventive action (percentage)
    max_memory_percent: float = 85.0
    
    # Fallback resolution when OOM occurs (reduced for low VRAM GPUs)
    fallback_width: int = 256
    fallback_height: int = 256
    
    # Number of inference steps to reduce on OOM retry
    reduction_steps: int = 5
    
    # Automatically cleanup memory before each generation
    auto_cleanup_enabled: bool = True
    
    # Maximum retries on OOM
    max_oom_retries: int = 3
    
    # Resolution reduction factor on each retry (0.75 = reduce to 75%)
    reduction_factor: float = 0.75


# Global configuration instances
device_config = DeviceConfig()
model_config = ModelConfig()
generation_config = GenerationConfig()
ui_config = UIConfig()
training_config = TrainingConfig()
memory_config = MemoryConfig()
