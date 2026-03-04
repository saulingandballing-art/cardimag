import runpod
import torch
import base64
import io
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

pipe = None

def load_model():
    global pipe
    model_id = "LyliaEngine/Pony_Diffusion_V6_XL"
    # Added 'device_map' and 'offload' to prevent exit code 1 memory crashes
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        variant="fp16",
        use_safetensors=True
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.enable_attention_slicing() # Saves massive VRAM

def handler(job):
    global pipe
    if pipe is None:
        load_model()
        
    job_input = job['input']
    prompt = job_input.get('prompt', 'score_9, score_8_up, a pony')
    
    # Generate smaller images to ensure the phone can handle the download
    image = pipe(
        prompt=f"score_9, score_8_up, {prompt}",
        num_inference_steps=20,
        width=512,
        height=512
    ).images[0]
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"image": img_str}

runpod.serverless.start({"handler": handler})
