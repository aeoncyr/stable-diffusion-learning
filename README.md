# 🎨 Stable Diffusion Image Editor

[![Tests](https://img.shields.io/badge/tests-passing-green)](tests)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A local image editing application using Stable Diffusion with **AMD Radeon (DirectML)** support.

## 🎓 Learning Project
This project is an educational resource to help you understand Generative AI engineering.
- **[TEXTBOOK (LEARNING.md)](docs/LEARNING.md)**: Deep dive into the Science (Physics, Math, U-Net Architecture).
- **In-Code Education**: Read comments blocks in the source code to learn more about that specific block.
- **Interactive Labs**: Check the `notebooks/` directory.

## Features
- 🎨 **Image-to-Image Editing**: Transform images with text prompts.
- 🖌️ **Inpainting**: Edit specific areas using a brush mask.
- ✨ **Text-to-Image**: Generate photorealistic images from scratch.
- 🔧 **LoRA Fine-Tuning**: Train custom models on your own data.
- 😊 **Face Restoration**: Built-in CodeFormer/GFPGAN support.
- 📊 **Live Memory Monitor**: Real-time VRAM usage tracking (New!).
- 🛡️ **OOM Protection**: Auto-recovery from "Out of Memory" errors.

## Installation

```bash
# 1. Clone/Navigate
cd "c:\Users\Advan Workplus\Projects\stable-diffusion"

# 2. Create Virtual Env
python -m venv venv
venv\Scripts\activate

# 3. Install
pip install -r requirements.txt
```

## Usage

### Run the App
```bash
python app.py
```
Open **http://localhost:7860** in your browser.

### Run Tests
Ensure reliability before making changes:
```bash
pip install pytest
pytest tests/
```

## Project Structure
| Directory | Description |
|-----------|-------------|
| `core/` | **Scientific Backend**: Pipeline, Diffusion Math, Memory Management |
| `ui/` | **Frontend**: Gradio components and tabs (Editor, Inpaint, Train) |
| `notebooks/` | **Interactive Labs**: Learn by doing (Jupyter) |
| `fine_tuning/` | **Training**: LoRA implementation logic |
| `tests/` | **Verification**: Unit tests for stability |

## 📚 Interactive Course (New!)
We have built a comprehensive **20-Lesson Curriculum** to take you from a beginner to an AI Engineer.

**👉 Start Here: [SYLLABUS.md](notebooks/SYLLABUS.md)**

### Modules
1.  **Foundations**: Diffusion basics, Tensors, Devices.
2.  **Architecture**: VAE, CLIP, U-Net, Schedulers.
3.  **Control**: CFG, Img2Img, Inpainting, Seeds.
4.  **Engineering**: Memory management, Optimization, Error handling.
5.  **Training**: LoRA fine-tuning theory and practice.
6.  **Capstone**: Building the Gradio App.


## Configuration
You can override standard paths using environment variables (useful for Docker/Cloud):
- `SD_MODELS_DIR`: Internal model cache
- `SD_OUTPUTS_DIR`: Output images
- `SD_DATASETS_DIR`: Training sources

## Contributing
We welcome contributions! Please run `pytest` before submitting a PR.
