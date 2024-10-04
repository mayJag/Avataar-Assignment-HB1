import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from torchvision import transforms

def setup_environment():
    """Set up the environment for model loading."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Make sure you have a GPU for better performance.")
    else:
        print("CUDA is available. Proceeding with GPU setup.")

def load_models():
    """Load the required models and return them."""
    print("Loading ControlNet and Stable Diffusion models...")

    # Load ControlNet model
    controlnet_model = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
    
    # Load Stable Diffusion ControlNet Pipeline
    stable_diffusion_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        controlnet=controlnet_model, 
        torch_dtype=torch.float16
    )
    
    # Enable GPU usage
    stable_diffusion_pipeline.to("cuda")

    print("Models loaded successfully.")
    return controlnet_model, stable_diffusion_pipeline

def apply_transform():
    """Apply the necessary image transformations."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

if __name__ == "__main__":
    setup_environment()
    controlnet, stable_diffusion_pipeline = load_models()
    transform = apply_transform()
