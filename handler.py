import runpod
import torch
import base64
import io
import logging
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

# Set up detailed logging to force errors into the RunPod console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipe = None

def load_model():
    global pipe
    logger.info("Starting model load...")
    try:
        model_id = "LyliaEngine/Pony_Diffusion_V6_XL"
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            variant="fp16", # Forces smaller file download
            use_safetensors=True
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")
        pipe.enable_xformers_memory_efficient_attention() # Saves VRAM
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"CRITICAL ERROR DURING MODEL LOAD: {e}")
        raise e

def handler(job):
    global pipe
    if pipe is None:
        load_model()
        
    job_input = job['input']
    prompt = job_input.get('prompt')
    
    image = pipe(
        prompt=prompt,
        negative_prompt=job_input.get('negative_prompt', ""),
        num_inference_steps=job_input.get('num_inference_steps', 25),
        guidance_scale=7.0,
        clip_skip=2
    ).images[0]
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {"image": img_str}

# Attempt load before starting the serverless listener
try:
    load_model()
except:
    pass

runpod.serverless.start({"handler": handler})
