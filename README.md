<h1 align="center" id="title">Avataar Assignment HB1</h1>

<p id="description">This project is an in-depth exploration of text-to-image generation using Stable Diffusion guided by ControlNet models, leveraging depth maps as inputs. The objective is to generate high-quality, photo-realistic images based on textual descriptions (prompts), conditioned by depth maps. We also experiment with generating images of different aspect ratios and evaluate the effect of varying inference steps on both image quality and latency.</p>

<p align="center"><img src="https://img.shields.io/badge/Python-3.8-blue" alt="shields"><img src="https://img.shields.io/badge/PyTorch-1.10+-orange" alt="shields"><img src="https://img.shields.io/badge/License-MIT-green" alt="shields"><img src="https://img.shields.io/badge/StableDiffusion-v1.5-purple" alt="shields"><img src="https://img.shields.io/badge/ControlNet-enabled-red" alt="shields"></p>

<h2>Project Screenshots:</h2>
<h3>Prompt: Minimalist room with a single chair and soft lighting</h3>
<h3>With 25 Steps</h3>
<img src="https://github.com/user-attachments/assets/eecdfc21-088a-453c-aeee-62d789e9002b" alt="project-screenshot"/>
<h3>With 25 vs 50 Steps</h3>
<img src="https://github.com/user-attachments/assets/37d5bbe5-fe1c-4a66-b4a0-4dbc38fcdd51" alt="project-screenshot"/>
<h3>Depth Map Given vs Depth Map extracted from Generated image</h3>
<img src="https://github.com/user-attachments/assets/892d9611-a02d-41cb-821d-0149bb61f2d0" alt="project-screenshot"/>
<h3>Experimented with different Schedulers(DDIM, LMS, PNDM)</h3>
<img src="https://github.com/user-attachments/assets/86082701-2fd8-4dd3-9d5a-6fdf0581cbea" alt="project-screenshot"/>
  
<h2>🧐 Features</h2>

Here're some of the project's best features:

*   Image generation based on text prompts and depth maps.
*   Comparison between square and non-square aspect ratios
*   Latency analysis and optimization.
*   ControlNet conditioning for better image guidance.
*   MSE Evaluation: Calculates Mean Squared Error (MSE) between input and generated depth maps for accuracy measurement.
*   Visual Analysis: Side-by-side comparison of generated images based on different conditions (aspect ratio number of steps).
*   Error Handling: Handles different types of input depth maps (e.g. .npy .png) and ensures safe normalization.
*   Experimentation with Different Schedulers: The project tests different inference schedulers like DDIM, PNDM, and others to optimize image quality and speed.
*   Visual Analysis: Displays side-by-side comparisons of generated images for different conditions and steps.

<h2>Flow of the Code:</h2>

<h3>User Input:</h3>

The user is prompted to select a text prompt from a list of predefined descriptions.
The user then selects a depth map, which can either be an image (.png) or a NumPy array (.npy) file.

<h3>Depth Map Loading:</h3>

The code checks if the selected depth map is in .npy format or an image format.
If it’s an .npy file, the depth map is loaded and normalized before being converted to an image.
If it’s an image, it is directly converted to RGB.

<h3>Aspect Ratio Check:</h3>

If a non-square depth map (5:3 aspect ratio) is selected, the code resizes the image accordingly (e.g., 940x564).
For square depth maps, the image is resized to 512x512 pixels.

<h3>Model Initialization:</h3>

The pre-trained ControlNet model with depth conditioning and the Stable Diffusion pipeline are loaded.
The pipeline is then transferred to the GPU for faster image generation.

<h3>Seed Fixing:</h3>

To ensure reproducibility of the generated images, a fixed seed is applied to the Stable Diffusion model.

<h3>Image Generation (25 Steps):</h3>

The selected text prompt and depth map are used to generate images with 25 inference steps.
The image generation time (latency) is recorded for analysis.

<h3>Comparison of 25 vs. 50 Steps:</h3>

After generating images with 25 steps, the process is repeated for 50 inference steps.
The generated images for both square and non-square depth maps are compared side by side.
The latency for 50 steps is also measured and compared with the latency of 25 steps.

<h3>Depth Map Estimation:</h3>

The code uses the MiDaS model to estimate the depth map from the generated images.
For both square and non-square images, the generated depth maps are compared with the input depth maps by calculating the Mean Squared Error (MSE).

<h3>Results Display:</h3>

The results, including the generated images and the depth map comparisons, are displayed side by side.
The MSE values are printed to assess the consistency between the input and generated depth maps.

<h3>Conclusion:</h3>

The project provides insights into the image generation process, the effect of inference steps on quality, and the flexibility of the model when handling different aspect ratios.


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
