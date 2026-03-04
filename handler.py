import runpod
import torch
import logging

# Simple logging to see what's happening
logging.basicConfig(level=logging.INFO)

def handler(job):
    # This is a heartbeat test
    return {"message": "System is alive. GPU detected: " + str(torch.cuda.is_available())}

# Start without loading the heavy model
runpod.serverless.start({"handler": handler})
