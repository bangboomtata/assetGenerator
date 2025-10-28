# Run this once to download base model components
from diffusers import DiffusionPipeline

# Download a compatible base model (you can do this once)
base_pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # or another compatible model
    cache_dir="./models"
)

# Save it locally
base_pipeline.save_pretrained("./models/base_model")