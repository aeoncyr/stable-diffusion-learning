
import sys
import os
from pathlib import Path
import torch
import warnings

# Suppress warnings for cleaner output in notebooks
warnings.filterwarnings("ignore")

def setup_notebook():
    """
    Sets up the notebook environment:
    1. Adds project root to sys.path
    2. Imports config
    3. Returns device and dtype
    """
    # Get project root (2 levels up from notebooks/)
    notebook_dir = Path(os.getcwd())
    project_root = notebook_dir.parent
    
    # Add to path if not exists
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Import config
    try:
        from config import device_config, model_config
        device = device_config.get_device()
        dtype = device_config.dtype
        print(f"✅ Setup Complete!")
        print(f"   - Project Root: {project_root}")
        print(f"   - Device: {device}")
        print(f"   - Dtype: {dtype}")
        return project_root, device, dtype
    except ImportError:
        print("❌ Could not import config. Make sure you are running this from the 'notebooks' folder.")
        return None, None, None

def show_image(image, size=(5, 5), title=None):
    """Helper to display PIL image with matplotlib"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=size)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Common imports users will need
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
