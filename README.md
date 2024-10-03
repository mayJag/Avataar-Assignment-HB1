<h1 align="center" id="title">Avataar Assignment HB1</h1>

<p id="description">This project is an in-depth exploration of text-to-image generation using Stable Diffusion guided by ControlNet models, leveraging depth maps as inputs. The objective is to generate high-quality, photo-realistic images based on textual descriptions (prompts), conditioned by depth maps. We also experiment with generating images of different aspect ratios and evaluate the effect of varying inference steps on both image quality and latency.</p>

<p align="center"><img src="https://img.shields.io/badge/Python-3.8-blue" alt="shields"><img src="https://img.shields.io/badge/PyTorch-1.10+-orange" alt="shields"><img src="https://img.shields.io/badge/License-MIT-green" alt="shields"><img src="https://img.shields.io/badge/StableDiffusion-v1.5-purple" alt="shields"><img src="https://img.shields.io/badge/ControlNet-enabled-red" alt="shields"></p>

<h2>Project Screenshots:</h2>
<h3>With 25 Steps</h3>
<img src="https://drive.google.com/uc?export=view&id=1aU4vApoWuNCVt-vOjFi9dUtfTpDClHkD" alt="project-screenshot" width="640" height="480" />
<h3>With 25 vs 50 Steps</h3>
<img src="https://drive.google.com/uc?export=view&id=1qDtSxQb5eBkWNITw4DaswVVjOkkvBw7e" alt="project-screenshot" width="640" height="480" />
<h3>Depth Map Given vs Depth Map Generated from Generated image</h3>
<img src="https://drive.google.com/uc?export=view&id=1sBUwDQmSeXy-PaMzw4gGGav0b5zhuByT" alt="project-screenshot" width="640" height="480" />
  
<h2>🧐 Features</h2>

Here're some of the project's best features:

*   Image generation based on text prompts and depth maps.
*   Comparison between square and non-square aspect ratios
*   Latency analysis and optimization.
*   ControlNet conditioning for better image guidance.
*   MSE Evaluation: Calculates Mean Squared Error (MSE) between input and generated depth maps for accuracy measurement.
*   Visual Analysis: Side-by-side comparison of generated images based on different conditions (aspect ratio number of steps).
*   Error Handling: Handles different types of input depth maps (e.g. .npy .png) and ensures safe normalization.

<h2>🛠️ Installation Steps:</h2>

<p>1. Step 1: Clone the repository</p>

```
git clone https://github.com/username/image-generation-project.git cd image-generation-project
```

<p>2. Step 2: Install required dependencies</p>

```
pip install -r requirements.txt
```

<p>3. Step 3: Set up the environment Ensure you have the necessary models from Huggingface or pre-trained checkpoints.</p>

```
python setup_model.py
```

<p>4. Step 4: Run the image generation script</p>

```
python generate_image.py --prompt "your prompt here" --depth_map "path_to_depth_map"
```

  
  
<h2>💻 Built with</h2>

Technologies used in the project:

*   PyTorch: For the core deep learning framework.
*   Stable Diffusion: Image generation model.
*   ControlNet: To guide generation with depth maps.
*   Matplotlib: For visualization.
