import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
import time
import matplotlib.pyplot as plt

# Load pre-trained ControlNet and Stable Diffusion models
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

# Enable GPU usage
pipe.to("cuda")

# Predefined prompt and depth map
prompt = "A beautiful landscape with mountains in the background."
depth_map_path = "C:/Users/jagga/PycharmProjects/FrugalTesting/Avtaar/Images/1.png"
depth_map_image = Image.open(depth_map_path).convert("RGB")
depth_map_image = depth_map_image.resize((512, 512))  # Rescale to 512x512

# Set the DDIM scheduler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Fix the seed for reproducibility
generator = torch.manual_seed(12345)

# Measure time for generation
start_time = time.time()

# Generate image
output = pipe(prompt=prompt, image=depth_map_image, generator=generator, num_inference_steps=25)

# Stop the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Save and display the generated image
output.images[0].save("generated_image_DDIM.png")
img = Image.open("generated_image_DDIM.png")
plt.imshow(img)
plt.title(f"Generated Image (DDIM Scheduler) - {elapsed_time:.2f} seconds")
plt.show()

# Print the elapsed time
print(f"DDIM scheduler took {elapsed_time:.2f} seconds.")
