"""
Memory Monitor UI Component
Provides real-time memory usage updates
"""

import gradio as gr
from typing import Tuple

from core.memory_manager import memory_manager, get_memory_status_string

def get_status_update() -> str:
    """Get updated memory status string"""
    return f"📊 {get_memory_status_string()}"

def create_memory_monitor() -> Tuple[gr.Markdown, gr.Timer]:
    """
    Create a memory monitor component with a timer for auto-updates.
    Returns: (status_markdown, timer)
    """
    status_display = gr.Markdown(value=f"📊 {get_memory_status_string()}")
    
    # Create a timer to update every 2 seconds
    timer = gr.Timer(value=2.0)
    
    timer.tick(fn=get_status_update, outputs=status_display)
        
    return status_display, timer
