# 📘 The Science of Stable Diffusion

Welcome to the deep dive! this document is a "textbook" designed to explain the rigorous mathematics and physics behind the code in this project.

## Table of Contents
1. [The Physics of Diffusion](#1-the-physics-of-diffusion)
2. [Latent Space Manifolds](#2-latent-space-manifolds)
3. [The Neural Architecture](#3-the-neural-architecture)
4. [Conditioning & Attention](#4-conditioning--attention)
5. [The Sampling Process](#5-the-sampling-process)

---

## 1. The Physics of Diffusion

### Thermodynamics and Entropy
Stable Diffusion is inspired by non-equilibrium thermodynamics.
Imagine a drop of ink falling into a glass of water. Over time, the ink diffuses until it is uniformly distributed. This is an increase in **Entropy** (disorder).
- **Forward Process**: Information $\to$ Noise (Easy, natural).
- **Reverse Process**: Noise $\to$ Information (Hard, requires energy/intelligence).

The model learns to reverse this process. It learns to "un-mix" the ink.

### The Math
We define a forward diffusion process $q$ that gradually adds Gaussian noise to data $x_0$ over $T$ steps:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I})$$

where $\beta_t$ is the variance schedule. As $T \to \infty$, $x_T$ becomes pure isotropic Gaussian noise.

The model trains a neural network $\epsilon_\theta(x_t, t)$ to predict the noise added at each step, effectively learning the reverse distribution $p_\theta(x_{t-1}|x_t)$.

---

## 2. Latent Space Manifolds

Why don't we diffuse pixels directly?
1. **Computational Cost**: A 512x512 image has 262,144 pixels. Calculating gradients for all of them is slow.
2. **Manifold Hypothesis**: Real-world data (faces, landscapes) lies on a lower-dimensional "manifold" embedded within the high-dimensional pixel space. Most random pixel combinations are just static.

### VAE (Variational Autoencoder)
We use a VAE to compress images into a **Latent Space**:
- **Encoder**: $\mathcal{E}(x) \to z$. Compresses 512x512x3 (786,432 values) $\to$ 64x64x4 (16,384 values). **48x compression!**
- **Decoder**: $\mathcal{D}(z) \to \hat{x}$. Reconstructs the image.

The diffusion happens in this compact latent space ($z$), making it faster and semantically richer.

---

## 3. The Neural Architecture

The core "brain" is a **U-Net**. It's a fully convolutional network with a specific shape:
1. **Downsampling (Encoder)**: Reduces spatial dimensions, increases feature depth. Captures "context".
2. **Bottleneck**: The most abstract representation of the image.
3. **Upsampling (Decoder)**: Increases spatial dimensions, reduces feature depth. Reconstructs "details".
4. **Skip Connections**: Direct links between Encoder and Decoder layers. Critical for preserving fine spatial details that would get lost in the bottleneck.

### ResBlocks
The building blocks are **ResBlocks** (Residual Blocks).
$$y = F(x) + x$$
The "skip connection" $+ x$ allows gradients to flow easily during training, preventing the vanishing gradient problem in deep networks.

---

## 4. Conditioning & Attention

How does the text prompt guide the image? Through **Cross-Attention**.

### CLIP Text Encoder
First, the text is converted to vectors by OpenAI's CLIP model.
- "A cat" $\to$ sequence of 768-dimensional vectors.

### Comparison to Self-Attention
- **Self-Attention**: The image looks at itself. "Is this leg part of this cat?"
- **Cross-Attention**: The image looks at the text. "Where should I put the 'cat' described in the prompt?"

### The Equation
$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **Query ($Q$)**: From the Image (U-Net features). "What am I looking for?"
- **Key ($K$)**: From the Text (CLIP). "What concepts do we have?"
- **Value ($V$)**: From the Text (CLIP). "Here is the concept feature."

---

## 5. The Sampling Process

Generating an image is solving a **Differential Equation** (DE).
The "noise predictor" effectively defines the gradient (slope) of the data distribution.

### The Probability Flow ODE
We travel along the trajectory from pure noise ($t=T$) to data ($t=0$).
$$dx = -\frac{1}{2} \beta(t) \left[ x + 2\nabla_{x} \log p_t(x) \right] dt$$

### Schedulers
Schedulers are solvers for this ODE.
- **Euler Ancestral**: Simple, fast, adds stochastic noise each step.
- **DDIM**: Deterministic, allows reversing the process (image $\to$ noise $\to$ image).
- **DPM++ 2M**: A high-order solver. Very accurate with few steps (20-30). Uses previous steps to estimate curvature, making it more efficient.

---

*This document serves as a scientific companion to the codebase. Check `config.py` and `core/pipeline.py` to see these formulas implemented!*
