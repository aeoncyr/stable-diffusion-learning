# 🎓 Stable Diffusion Learning Journey: Syllabus

Welcome to your comprehensive course on Stable Diffusion engineering! This curriculum is split into **20 focused notebooks**, designed to take you from a beginner to an advanced AI engineer capable of building your own applications.

## 🟢 Module 1: The Basics (Foundations)
1.  **[01_concept_of_diffusion.ipynb](01_concept_of_diffusion.ipynb)**
    *   What is diffusion? The "Forward" vs "Reverse" process.
    *   Gaussian noise math.
    *   Loading your first Pipeline.
2.  **[02_tensors_and_devices.ipynb](02_tensors_and_devices.ipynb)**
    *   PyTorch basics (if you are new).
    *   Understanding GPUs, VRAM, and FP16 vs FP32 precision.
    *   Why we use `DirectML` for AMD.
3.  **[03_first_generation.ipynb](03_first_generation.ipynb)**
    *   Running `txt2img`.
    *   Anatomy of the output (PIL Images, seeds).
    *   Basic prompt engineering.

## 🔵 Module 2: Architecture Deep Dive (The "Engine")
4.  **[04_vae_and_latent_space.ipynb](04_vae_and_latent_space.ipynb)**
    *   The Manifold Hypothesis.
    *   Why 512x512 is actually 64x64.
    *   Visualizing the VAE Compressor/Decompressor.
5.  **[05_clip_and_tokenization.ipynb](05_clip_and_tokenization.ipynb)**
    *   How computers read English.
    *   Tokens, Embeddings, and Vectors.
    *   The 77-token limit explained.
6.  **[06_the_unet_predictor.ipynb](06_the_unet_predictor.ipynb)**
    *   Inside the massive "Noise Predictor".
    *   ResNets and Attention mechanisms.
    *   Connecting Text to Image (Cross-Attention).
7.  **[07_schedulers_explained.ipynb](07_schedulers_explained.ipynb)**
    *   Solving Differential Equations.
    *   Euler vs DDIM vs DPM++.
    *   Step count impact analysis.

## 🟠 Module 3: Controlling Generation (The "Steering Wheel")
8.  **[08_guidance_math_cfg.ipynb](08_guidance_math_cfg.ipynb)**
    *   Classifier-Free Guidance (CFG).
    *   The Vector Math of "Listening to the Prompt".
    *   Visualizing the "Fry" effect of high guidance.
9.  **[09_image_to_image_math.ipynb](09_image_to_image_math.ipynb)**
    *   The concept of SDEdit.
    *   Noise Injection Strength (0.0 to 1.0).
    *   Denoising from a starting point.
10. **[10_inpainting_masks.ipynb](10_inpainting_masks.ipynb)**
    *   How masks work (Binary latents).
    *   The VAE-Masking problem.
    *   Blurring and blending edges.
11. **[11_seeds_and_determinism.ipynb](11_seeds_and_determinism.ipynb)**
    *   Chaotic systems.
    *   Why 1 bit of difference changes everything.
    *   Building reproducible pipelines.

## 🔴 Module 4: Engineering & Optimization (The "Mechanics")
12. **[12_memory_management_101.ipynb](12_memory_management_101.ipynb)**
    *   Understanding VRAM allocation.
    *   Garbage Collection (`gc.collect`).
    *   The cost of model swapping.
13. **[13_optimization_techniques.ipynb](13_optimization_techniques.ipynb)**
    *   Attention Slicing (trading time for memory).
    *   VAE Tiling (handling 4K images).
    *   CPU Offloading.
14. **[14_building_robust_pipelines.ipynb](14_building_robust_pipelines.ipynb)**
    *   Error handling (OOM protection).
    *   Abstracting complexity (The `PipelineManager` class).

## 🟣 Module 5: Training & Customization (The "Laboratory")
15. **[15_lora_theory.ipynb](15_lora_theory.ipynb)**
    *   Low-Rank Adaptation math ($W = A \times B$).
    *   Why files are small (100MB vs 4GB).
16. **[16_dataset_preparation.ipynb](16_dataset_preparation.ipynb)**
    *   Image preprocessing (Aspect ratios, bucketing).
    *   Captioning strategies.
17. **[17_training_loop_walkthrough.ipynb](17_training_loop_walkthrough.ipynb)**
    *   The specific training code in `lora_trainer.py`.
    *   Gradients, Optimizers (AdamW), and Loss.

## ⚫ Module 6: Building the App (Capstone)
18. **[18_gradio_basics.ipynb](18_gradio_basics.ipynb)**
    *   Events, Inputs, Outputs.
    *   Blocks vs Interface.
19. **[19_state_management.ipynb](19_state_management.ipynb)**
    *   Global state vs Session state.
    *   Queue management.
20. **[20_project_capstone.ipynb](20_project_capstone.ipynb)**
    *   Reviewing the full `app.py`.
    *   Future directions (ControlNet, SDXL).
