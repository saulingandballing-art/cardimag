import runpod
import torch
import base64
import io
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

# 1. Define the HuggingFace repo for Pony V6 XL
MODEL_ID = "LyliaEngine/Pony_Diffusion_V6_XL" 

# 2. Load the model OUTSIDE the handler. 
# This keeps the model in VRAM between requests so "warm" generations take 3 seconds instead of 30.
print("Loading Pony V6 XL into VRAM...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    use_safetensors=True
)

# Pony works best with Euler A
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
print("Model loaded successfully.")

# 3. The Handler function that Runpod triggers when Kivy sends a request
def handler(job):
    # Extract the payload sent from your Pydroid app
    job_input = job['input']
    
    prompt = job_input.get('prompt', 'score_9, score_8_up, a beautiful landscape')
    neg_prompt = job_input.get('negative_prompt', '')
    steps = job_input.get('num_inference_steps', 25)
    cfg = job_input.get('guidance_scale', 7.0)
    
    # Generate the image
    image = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        clip_skip=2 # Pony strictly requires Clip Skip 2 to avoid blurry blobs
    ).images[0]
    
    # Convert PIL Image directly to Base64 in RAM
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Return the exact JSON structure Kivy is expecting
    return {"image": img_str}

# 4. Start the Runpod serverless listener
runpod.serverless.start({"handler": handler})
