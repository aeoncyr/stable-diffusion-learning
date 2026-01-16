"""
Memory Manager Module
OOM prevention and memory optimization for Stable Diffusion

Provides:
- Memory monitoring utilities
- Automatic garbage collection
- OOM-protected decorator for safe generation
- Progressive degradation on memory pressure
"""

import gc
import functools
import traceback
from typing import Optional, Tuple, Callable, Any
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class MemoryStatus:
    """Current memory status"""
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    vram_percent: Optional[float] = None
    device_name: str = "Unknown"
    
    def __str__(self) -> str:
        status = f"RAM: {self.ram_used_gb:.1f}/{self.ram_total_gb:.1f}GB ({self.ram_percent:.0f}%)"
        if self.vram_total_gb is not None:
            status += f" | VRAM: {self.vram_used_gb:.1f}/{self.vram_total_gb:.1f}GB ({self.vram_percent:.0f}%)"
        return status


class MemoryManager:
    """
    Manages VRAM usage to prevent Out Of Memory (OOM) errors
    
    === VRAM Management ===
    Running AI models locally requires careful management of Video RAM.
    If we load too many models or process images that are too large, the GPU crashes.
    
    === GPU Memory Management ===
    Managing VRAM (Video RAM) is critical for local AI.
    - **Dedicated VRAM**: High-speed memory on the GPU (e.g., 6GB, 8GB).
    - **Shared Memory**: System RAM used when VRAM is full (much slower).
    
    This manager attempts to prevent "Out Of Memory" (OOM) errors by:
    1. **Garbage Collection (GC)**: Python's automatic memory cleaner.
    2. **Torch Cache Clearing**: `torch.cuda.empty_cache()` releases unused reserved memory back to the OS.
    3. **Aggressive Cleanup**: Forcefully deleting models when memory is dangerously low.
    This class tracks memory usage and clears caches (temporary data) when needed.
    """
    
    def __init__(self):
        self._last_status: Optional[MemoryStatus] = None
        
    def get_memory_status(self) -> MemoryStatus:
        """Get current memory status for RAM and VRAM"""
        import psutil
        
        # RAM info
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024 ** 3)
        ram_total_gb = ram.total / (1024 ** 3)
        ram_percent = ram.percent
        
        # VRAM info
        vram_used_gb = None
        vram_total_gb = None
        vram_percent = None
        device_name = "CPU"
        
        # Try CUDA first
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                vram_used_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
                vram_percent = (vram_used_gb / vram_total_gb) * 100 if vram_total_gb > 0 else 0
        except Exception:
            pass
        
        # Try DirectML (limited info available)
        if vram_total_gb is None:
            try:
                import torch_directml
                device_name = "AMD GPU (DirectML)"
                # DirectML doesn't expose memory stats directly
                # We'll estimate based on system memory pressure
            except ImportError:
                pass
        
        status = MemoryStatus(
            ram_used_gb=ram_used_gb,
            ram_total_gb=ram_total_gb,
            ram_percent=ram_percent,
            vram_used_gb=vram_used_gb,
            vram_total_gb=vram_total_gb,
            vram_percent=vram_percent,
            device_name=device_name
        )
        
        self._last_status = status
        return status
    
    def cleanup_memory(self, aggressive: bool = False) -> None:
        """
        Clean up memory
        
        Args:
            aggressive: If True, performs more thorough cleanup
        """
        # Python garbage collection
        gc.collect()
        
        # PyTorch CUDA cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                if aggressive:
                    torch.cuda.synchronize()
        except Exception:
            pass
        
        # Additional garbage collection passes for aggressive mode
        if aggressive:
            for _ in range(3):
                gc.collect()
        
        print("✓ Memory cleanup completed")
    
    def check_memory_available(
        self, 
        min_ram_gb: float = 2.0,
        min_ram_percent_free: float = 20.0
    ) -> Tuple[bool, str]:
        """
        Check if enough memory is available for generation
        
        Returns:
            Tuple of (is_available, message)
        """
        status = self.get_memory_status()
        
        ram_free_gb = status.ram_total_gb - status.ram_used_gb
        ram_free_percent = 100 - status.ram_percent
        
        if ram_free_gb < min_ram_gb:
            return False, f"Low RAM: {ram_free_gb:.1f}GB free (need {min_ram_gb:.1f}GB)"
        
        if ram_free_percent < min_ram_percent_free:
            return False, f"RAM nearly full: {ram_free_percent:.0f}% free"
        
        # Check VRAM if available
        if status.vram_percent is not None and status.vram_percent > 90:
            return False, f"VRAM nearly full: {status.vram_percent:.0f}% used"
        
        return True, "Memory OK"
    
    def estimate_memory_requirement(
        self,
        width: int,
        height: int,
        batch_size: int = 1,
        use_fp16: bool = True
    ) -> float:
        """
        Estimate memory requirement in GB for generation
        
        This is a rough estimate based on typical SD 1.5 usage
        """
        # Base model size (compressed)
        base_memory = 2.0 if use_fp16 else 4.0
        
        # Image tensor size estimate
        # Each pixel needs ~16 bytes per channel with gradients
        pixels = width * height * batch_size
        image_memory = (pixels * 16 * 4) / (1024 ** 3)  # 4 channels, 16 bytes
        
        # Scale for step overhead
        total = base_memory + image_memory * 3  # Triple for intermediate tensors
        
        return total
    
    def get_safe_resolution(
        self,
        target_width: int,
        target_height: int,
        max_memory_gb: float = 6.0
    ) -> Tuple[int, int]:
        """
        Get a safe resolution that should fit in memory
        
        Args:
            target_width: Desired width
            target_height: Desired height
            max_memory_gb: Maximum memory to target
            
        Returns:
            Tuple of (width, height) that should be safe
        """
        required = self.estimate_memory_requirement(target_width, target_height)
        
        if required <= max_memory_gb:
            return target_width, target_height
        
        # Calculate scale factor needed
        scale = (max_memory_gb / required) ** 0.5
        
        new_width = int(target_width * scale)
        new_height = int(target_height * scale)
        
        # Round down to multiple of 8
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        # Minimum size
        new_width = max(new_width, 256)
        new_height = max(new_height, 256)
        
        return new_width, new_height


# Singleton instance
memory_manager = MemoryManager()


def oom_protected(
    max_retries: int = 3,
    cleanup_before: bool = True,
    fallback_resolution: Tuple[int, int] = (384, 384),
    reduction_factor: float = 0.75
):
    """
    Decorator that provides OOM protection for generation functions
    
    Args:
        max_retries: Maximum retry attempts on OOM
        cleanup_before: Run cleanup before execution
        fallback_resolution: Ultimate fallback resolution
        reduction_factor: Factor to reduce resolution on each retry
    
    Usage:
        @oom_protected(max_retries=3)
        def generate_image(prompt, width=512, height=512, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            from config import memory_config
            
            if not memory_config.enable_oom_protection:
                return func(*args, **kwargs)
            
            # Clean up before if enabled
            if cleanup_before and memory_config.auto_cleanup_enabled:
                memory_manager.cleanup_memory()
            
            last_error = None
            current_width = kwargs.get('width')
            current_height = kwargs.get('height')
            current_steps = kwargs.get('num_steps')
            
            for attempt in range(max_retries + 1):
                try:
                    # Check memory before attempting
                    is_ok, msg = memory_manager.check_memory_available()
                    if not is_ok:
                        print(f"⚠ Memory warning: {msg}")
                        memory_manager.cleanup_memory(aggressive=True)
                    
                    result = func(*args, **kwargs)
                    
                    # Success - return result
                    if attempt > 0:
                        print(f"✓ Generation succeeded on retry {attempt}")
                    return result
                    
                except (RuntimeError, MemoryError) as e:
                    error_msg = str(e).lower()
                    
                    # Check if this is an OOM error (including DirectML-specific messages)
                    is_oom = any(phrase in error_msg for phrase in [
                        'out of memory',
                        'cuda out of memory',
                        'oom',
                        'allocate',
                        'memory',
                        'directml',
                        'dml allocator',
                        'gpu will not respond',
                        'invalid commands',
                        'device memory'
                    ])
                    
                    if not is_oom:
                        # Not an OOM error, re-raise
                        raise
                    
                    last_error = e
                    print(f"\n⚠ OOM Error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    
                    # Clean up aggressively
                    memory_manager.cleanup_memory(aggressive=True)
                    
                    if attempt >= max_retries:
                        break
                    
                    # Reduce parameters for next attempt
                    if current_width and current_height:
                        new_width = int(current_width * reduction_factor)
                        new_height = int(current_height * reduction_factor)
                        
                        # Round to multiple of 8
                        new_width = max((new_width // 8) * 8, fallback_resolution[0])
                        new_height = max((new_height // 8) * 8, fallback_resolution[1])
                        
                        if new_width < current_width or new_height < current_height:
                            print(f"  → Reducing resolution: {current_width}x{current_height} → {new_width}x{new_height}")
                            kwargs['width'] = new_width
                            kwargs['height'] = new_height
                            current_width = new_width
                            current_height = new_height
                    
                    # Reduce steps if possible
                    if current_steps and current_steps > 10:
                        new_steps = max(current_steps - memory_config.reduction_steps, 10)
                        if new_steps < current_steps:
                            print(f"  → Reducing steps: {current_steps} → {new_steps}")
                            kwargs['num_steps'] = new_steps
                            current_steps = new_steps
                    
                    print(f"  → Retrying with reduced settings...")
            
            # All retries failed
            error_message = (
                f"Generation failed due to Out of Memory after {max_retries + 1} attempts.\n\n"
                f"Suggestions to reduce memory usage:\n"
                f"1. Use smaller image size (384x384 or 256x256)\n"
                f"2. Enable CPU Offload in Settings\n"
                f"3. Close other applications\n"
                f"4. Reduce inference steps\n"
                f"5. Clear model cache in Settings\n\n"
                f"Original error: {last_error}"
            )
            raise RuntimeError(error_message)
        
        return wrapper
    return decorator


def get_memory_status_string() -> str:
    """Get a formatted memory status string for UI display"""
    status = memory_manager.get_memory_status()
    return str(status)


def pre_generation_cleanup() -> None:
    """Standard cleanup to run before generation"""
    from config import memory_config
    
    if memory_config.auto_cleanup_enabled:
        memory_manager.cleanup_memory()
