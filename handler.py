import runpod
import torch
import base64
import io
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

pipe = None

def load_model():
    global pipe
    model_id = "LyliaEngine/Pony_Diffusion_V6_XL"
    # Load with fp16 to save 50% VRAM and disk space
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        variant="fp16",
        use_safetensors=True
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    # This is the secret sauce for speed and memory efficiency
    pipe.enable_xformers_memory_efficient_attention() 

def handler(job):
    global pipe
    if pipe is None:
        load_model()
        
    job_input = job['input']
    prompt = job_input.get('prompt')
    
    # Pony V6 XL works best with 'score_9, score_8_up' in the prompt
    image = pipe(
        prompt=prompt,
        negative_prompt=job_input.get('negative_prompt', "lowres, bad anatomy, text, error"),
        num_inference_steps=job_input.get('num_inference_steps', 25),
        guidance_scale=7.0
    ).images[0]
    
    # Convert the image to a Base64 string for the app to read
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {"image": img_str}

runpod.serverless.start({"handler": handler})
