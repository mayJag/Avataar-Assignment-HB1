<h1 align="center" id="title">Avataar Assignment HB1</h1>

<p id="description">This project is an in-depth exploration of text-to-image generation using Stable Diffusion guided by ControlNet models, leveraging depth maps as inputs. The objective is to generate high-quality, photo-realistic images based on textual descriptions (prompts), conditioned by depth maps. We also experiment with generating images of different aspect ratios and evaluate the effect of varying inference steps on both image quality and latency.</p>

<p align="center"><img src="https://img.shields.io/badge/Python-3.8-blue" alt="shields"><img src="https://img.shields.io/badge/PyTorch-1.10+-orange" alt="shields"><img src="https://img.shields.io/badge/License-MIT-green" alt="shields"><img src="https://img.shields.io/badge/StableDiffusion-v1.5-purple" alt="shields"><img src="https://img.shields.io/badge/ControlNet-enabled-red" alt="shields"></p>

<h2>Project Screenshots:</h2>
<h3>Prompt: Minimalist room with a single chair and soft lighting</h3>
<h3>With 25 Steps</h3>
<img src="https://github.com/user-attachments/assets/eecdfc21-088a-453c-aeee-62d789e9002b" alt="project-screenshot"/>
<h3>With 25 vs 50 Steps</h3>
<img src="https://github.com/user-attachments/assets/37d5bbe5-fe1c-4a66-b4a0-4dbc38fcdd51" alt="project-screenshot"/>
<h3>Depth Map Given vs Depth Map extracted from Generated image(used MSE and SSIM for Evaluating)</h3>
<img src="https://github.com/user-attachments/assets/892d9611-a02d-41cb-821d-0149bb61f2d0" alt="project-screenshot"/>
<h3>Experimented with different Schedulers(DDIM, LMS, PNDM)</h3>
<img src="https://github.com/user-attachments/assets/86082701-2fd8-4dd3-9d5a-6fdf0581cbea" alt="project-screenshot"/>
  
<h2>🧐 Features</h2>

Here're some of the project's best features:

*   Image generation based on text prompts and depth maps.
*   Comparison between square and non-square aspect ratios
*   Latency analysis and optimization.
*   ControlNet conditioning for better image guidance.
*   MSE Evaluation: Calculates Mean Squared Error (MSE) and Structural Similarity Index(SSIM) between input and generated depth maps for accuracy measurement.
*   Visual Analysis: Side-by-side comparison of generated images based on different conditions (aspect ratio number of steps).
*   Error Handling: Handles different types of input depth maps (e.g. .npy .png) and ensures safe normalization.
*   Experimentation with Different Schedulers: The project tests different inference schedulers like DDIM, PNDM, and others to optimize image quality and speed.
*   Visual Analysis: Displays side-by-side comparisons of generated images for different conditions and steps.

<h2>Tasks- </h2>
1. For the given metadata, i.e., text description and depths, generate the “best” possible output images.
<img src="https://github.com/user-attachments/assets/fec0cfbc-9ffe-47f8-a604-dbd52e760738" alt="project-screenshot"/>

2. Can we generate images of different aspect ratios (use “Metadata/No crop/2_nocrop.png” to test this out) using SD? Comment on the generation quality with respect to the aspect ratio of 1:1 for the same image.

<img src="https://github.com/user-attachments/assets/7d049065-2ffb-4b38-bfb7-09b92c82cf40" alt="project-screenshot"/>

For the square aspect ratio (1:1) image, the composition feels more centered and balanced, making it easier to focus on the main subject (the bed). The detail is more concentrated due to the uniform frame.

In contrast, the non-square (5:3) image offers a wider field of view, capturing more of the environment and context but can make the main subject feel less prominent. The extra space might dilute focus on finer details compared to the square version.

3. What is the generation latency? Do you see some quick fixes to reduce it? Comment on how much latency you can reduce. What happens to the generation quality with reduced latency?

<img src="https://github.com/user-attachments/assets/1bc6e6d7-48ee-4ecf-8cce-52e7c43dd5c7" alt="project-screenshot"/>

<h3>Steps taken to reduce Genration Latency-</h3>

1. DDIM scheduler: Allows for faster image generation with fewer steps.

2. Reduced inference steps (25): Minimizes the number of denoising steps.

3. GPU usage: Accelerates model inference and depth map estimation.

4. Fixed seed: Ensures consistent results without extra variability.

5. Efficient image preprocessing: Resizing images to match model requirements.

6. Disabling gradient computation: Saves unnecessary computation during depth map prediction.

7. Pre-loading and mixed precision models: Ensures readiness and efficiency in model computation.

<h3>Generation quality with reduced latency-</h3>

In general, with 25 steps, image generation is faster but may result in less sharpness, softer details, and some artifacts. It’s suitable for quick results but lacks finer textures. With 50 steps, the image becomes more refined, with sharper details, better textures, and increased realism, though it takes longer to generate.

<h3>Guidelines Followed - </h3>

1. Used the given Checkpoint - https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5✅

2. Kept seed fixed to "12345"✅

3. For the generated outputs, verified that the generated depths are the same as the input depths using metrics MSE and SSIM.✅

4. The pipeline/heuristic working consistently across all the images. Did not “manually” change any metadata.✅


<h2>Flow of the Code:</h2>

<img src="https://github.com/user-attachments/assets/dffda710-02d9-4461-85e5-11243ae18bf4" alt="project-screenshot" width="400" height="400"/>

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
python main.py --prompt "your prompt here" --depth_map "path_to_depth_map"
```

  
  
<h2>💻 Built with</h2>

Technologies used in the project:

*   PyTorch: For the core deep learning framework.
*   Stable Diffusion: Image generation model.
*   ControlNet: To guide generation with depth maps.
*   Matplotlib: For visualization.
