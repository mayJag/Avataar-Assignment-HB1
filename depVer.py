import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
import torchvision.transforms as transforms
import cv2

# Safe normalization function for depth map
def safe_normalize(depth_map):
    """Normalize the depth map to the range [0, 1]."""
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)
    if max_val > min_val:
        normalized_map = (depth_map - min_val) / (max_val - min_val)
    else:
        normalized_map = np.zeros_like(depth_map)
    return normalized_map

# Function to load MiDaS model
def load_midas_model():
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas.eval()
    if torch.cuda.is_available():
        midas = midas.to(torch.device("cuda"))
    return midas

# Function to apply the necessary transformations
def apply_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

# Function to estimate depth from an image
def estimate_depth(image_path, model, transform):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image from {image_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image to 512x512, consistent with model input size
    img = cv2.resize(img, (512, 512))

    input_img = transform(img).unsqueeze(0)
    input_img = input_img.to(next(model.parameters()).device)

    with torch.no_grad():
        prediction = model(input_img)

    prediction = prediction.squeeze().cpu().numpy()

    return prediction


# Predefined prompts
prompts = [
    "beautiful landscape, mountains in the background",
    "luxury bedroom interior",
    "beautiful snowy mountains",
    "luxurious bedroom interior",
    "walls with cupboard",
    "room with chair",
    "house in the forest",
    "Lush green valley with mountain peaks in the distance",
    "Cozy bedroom with large windows and natural light",
    "Snow-covered pine trees against mountain backdrops",
    "Modern luxury living room with large glass walls",
    "A cozy room featuring aged, rustic wooden log texture walls and ornate cupboards filled with vintage decor.",
    "Minimalist room with a single chair and soft lighting",
    "Cabin nestled deep in the forest, surrounded by tall trees",
    "Sunset view over a lake with mountains in the background"
]

# Available depth maps
depth_files = [
    "C:/Users/jagga/PycharmProjects/FrugalTesting/Avtaar/Images/1.png",  # Image file
    "C:/Users/jagga/PycharmProjects/FrugalTesting/Avtaar/Images/2.png",  # Image file
    "C:/Users/jagga/PycharmProjects/FrugalTesting/Avtaar/Images/3.png",  # Image file
    "C:/Users/jagga/PycharmProjects/FrugalTesting/Avtaar/Images/4.png",  # Image file
    "C:/Users/jagga/PycharmProjects/FrugalTesting/Avtaar/Images/5.png",  # Image file
    "C:/Users/jagga/PycharmProjects/FrugalTesting/Avtaar/Images/6.png",  # Image file
    "C:/Users/jagga/PycharmProjects/FrugalTesting/Avtaar/Images/7.png",  # Image file
    "C:/Users/jagga/PycharmProjects/FrugalTesting/Avtaar/Images/6.npy",  # NPY file
    "C:/Users/jagga/PycharmProjects/FrugalTesting/Avtaar/Images/7.npy",   # NPY file
    "C:/Users/jagga/PycharmProjects/FrugalTesting/Avtaar/No Crop/2_nocrop.png"  # Non-square image (940x564)
]

# Ask user to select a prompt
print("Select one of the following prompts:")
for idx, prompt in enumerate(prompts, 1):
    print(f"{idx}: {prompt}")
prompt_idx = int(input("Enter the number of your chosen prompt: ")) - 1
selected_prompt = prompts[prompt_idx]

# Ask user to select a depth map
print("\nSelect a depth map (image or .npy file):")
for idx, depth_file in enumerate(depth_files, 1):
    print(f"{idx}: {depth_file}")
depth_idx = int(input("Enter the number of your chosen depth map: ")) - 1
selected_depth_file = depth_files[depth_idx]

# Check if the depth map is an .npy file or image file
if selected_depth_file.endswith(".npy"):
    # Load depth map from .npy file
    depth_map = np.load(selected_depth_file)
    depth_map = safe_normalize(depth_map)  # Normalize it
    depth_map_image = Image.fromarray((depth_map * 255).astype(np.uint8))  # Convert to PIL image
    depth_map_image = depth_map_image.convert("RGB")
else:
    # Load depth map from image file
    depth_map_image = Image.open(selected_depth_file).convert("RGB")


# Load pre-trained ControlNet and Stable Diffusion models
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

# Use DDIM scheduler for faster inference with lower steps
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Enable GPU usage
pipe.to("cuda")

# Fix the seed for reproducibility
generator = torch.manual_seed(12345)


# Start measuring time
start_time = time.time()

# Check if the selected depth map is non-square or square
if selected_depth_file.endswith("nocrop.png"):
    # Non-square depth map (Aspect ratio 940x564)
    depth_map_non_square = Image.open(selected_depth_file).convert("RGB")
    depth_map_non_square = depth_map_non_square.resize((940, 564))  # Keep the 5:3 aspect ratio
    
    # Load square version for comparison
    depth_map_square = Image.open("C:/Users/jagga/PycharmProjects/FrugalTesting/Avtaar/Images/2.png").convert("RGB")
    depth_map_square = depth_map_square.resize((512, 512))  # Resize to 1:1 aspect ratio

    # Generate images for both non-square and square depth maps
    output_non_square = pipe(prompt=selected_prompt, image=depth_map_non_square, generator=generator, num_inference_steps=25)
    output_square = pipe(prompt=selected_prompt, image=depth_map_square, generator=generator, num_inference_steps=25)

    # Save and display the generated images
    output_non_square.images[0].save("generated_image_non_square.png")
    output_square.images[0].save("generated_image_square.png")

    # Show the images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    img_non_square = Image.open("generated_image_non_square.png")
    img_square = Image.open("generated_image_square.png")

    axes[0].imshow(img_non_square)
    axes[0].set_title("Non-Square Image (5:3)")

    axes[1].imshow(img_square)
    axes[1].set_title("Square Image (1:1)")

    plt.show()
else:
    # Square depth map (512x512)
    depth_map = depth_map_image.resize((512, 512))  # Rescale to 512x512

    # Generate the image using Stable Diffusion
    output = pipe(prompt=selected_prompt, image=depth_map, generator=generator, num_inference_steps=25)

    # Save and display the generated image
    output.images[0].save("generated_image.png")
    img = Image.open("generated_image.png")
    plt.imshow(img)
    plt.title("Generated Image (Square 1:1)")
    plt.show()

# Stop time measurement and display
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Image generation completed in {elapsed_time:.2f} seconds.")

# Compare the times and images of 25 and 50 steps
print("\nComparing 25 steps vs 50 steps...")

# Measure time for 25 steps (already done above)
time_25_steps = elapsed_time

# Start measuring time for 50 steps
start_time_50 = time.time()

# Generate images with 50 steps
if selected_depth_file.endswith("nocrop.png"):
    # Non-square depth map (Aspect ratio 940x564)
    output_non_square_50 = pipe(prompt=selected_prompt, image=depth_map_non_square, generator=generator, num_inference_steps=50)
    output_square_50 = pipe(prompt=selected_prompt, image=depth_map_square, generator=generator, num_inference_steps=50)

    # Save and display the generated images for 50 steps
    output_non_square_50.images[0].save("generated_image_non_square_50_steps.png")
    output_square_50.images[0].save("generated_image_square_50_steps.png")

    # Show the 50-step images side by side with the 25-step images
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    img_non_square_50 = Image.open("generated_image_non_square_50_steps.png")
    img_square_50 = Image.open("generated_image_square_50_steps.png")

    axes[0, 0].imshow(img_non_square)
    axes[0, 0].set_title("Non-Square Image (25 Steps)")

    axes[0, 1].imshow(img_square)
    axes[0, 1].set_title("Square Image (25 Steps)")

    axes[1, 0].imshow(img_non_square_50)
    axes[1, 0].set_title("Non-Square Image (50 Steps)")

    axes[1, 1].imshow(img_square_50)
    axes[1, 1].set_title("Square Image (50 Steps)")

    plt.show()

else:
    # Square depth map (512x512)
    depth_map = depth_map_image.resize((512, 512))  # Rescale to 512x512

    # Generate the image using Stable Diffusion with 50 steps
    output_50 = pipe(prompt=selected_prompt, image=depth_map, generator=generator, num_inference_steps=50)

    # Save and display the generated image for 50 steps
    output_50.images[0].save("generated_image_50_steps.png")

    img_50 = Image.open("generated_image_50_steps.png")

    # Show the 25-step and 50-step images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img)
    axes[0].set_title("Generated Image (25 Steps)")

    axes[1].imshow(img_50)
    axes[1].set_title("Generated Image (50 Steps)")

    plt.show()

# Stop time measurement for 50 steps and display
end_time_50 = time.time()
time_50_steps = end_time_50 - start_time_50

# Print the time comparison
print(f"Image generation completed in {time_25_steps:.2f} seconds (25 steps).")
print(f"Image generation completed in {time_50_steps:.2f} seconds (50 steps).")

# Depth map comparison for non-square image
midas_model = load_midas_model()
transform = apply_transform()

if selected_depth_file.endswith("nocrop.png"):
    # Estimate depth map for the non-square generated image
    generated_depth_map_non_square = estimate_depth("generated_image_non_square.png", midas_model, transform)

    if generated_depth_map_non_square is not None:
        input_depth_map_non_square = np.asarray(depth_map_non_square)
        if input_depth_map_non_square.shape[2] == 3:
            input_depth_map_non_square = input_depth_map_non_square[:, :, 0]  # Convert to single channel

        generated_depth_map_non_square_resized = cv2.resize(generated_depth_map_non_square, 
            (input_depth_map_non_square.shape[1], input_depth_map_non_square.shape[0]))

        input_depth_map_non_square_normalized = safe_normalize(input_depth_map_non_square)
        generated_depth_map_non_square_normalized = safe_normalize(generated_depth_map_non_square_resized)

        # Display depth maps side by side for non-square comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(input_depth_map_non_square, cmap="gray")
        axes[0].set_title("Input Depth Map (Non-Square)")
        axes[1].imshow(generated_depth_map_non_square_resized, cmap="gray")
        axes[1].set_title("Generated Depth Map (Non-Square)")
        plt.show()

        if input_depth_map_non_square_normalized.shape == generated_depth_map_non_square_normalized.shape:
            mse_value_non_square = mean_squared_error(input_depth_map_non_square_normalized, generated_depth_map_non_square_normalized)
            print(f"MSE (Non-Square): {mse_value_non_square:.4f}")

# Depth map comparison for square image
generated_depth_map = estimate_depth("generated_image.png", midas_model, transform)

if generated_depth_map is not None:
    input_depth_map = np.asarray(depth_map)
    if input_depth_map.shape[2] == 3:
        input_depth_map = input_depth_map[:, :, 0]  # Convert to single channel

    generated_depth_map_resized = cv2.resize(generated_depth_map, (input_depth_map.shape[1], input_depth_map.shape[0]))

    input_depth_map_normalized = safe_normalize(input_depth_map)
    generated_depth_map_normalized = safe_normalize(generated_depth_map_resized)

    # Display depth maps side by side for square comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(input_depth_map, cmap="gray")
    axes[0].set_title("Input Depth Map (Square)")
    axes[1].imshow(generated_depth_map_resized, cmap="gray")
    axes[1].set_title("Generated Depth Map (Square)")
    plt.show()

    if input_depth_map_normalized.shape == generated_depth_map_normalized.shape:
        mse_value_square = mean_squared_error(input_depth_map_normalized, generated_depth_map_normalized)
        print(f"MSE (Square): {mse_value_square:.4f}")
