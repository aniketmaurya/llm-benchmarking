import lightning as L
import torch
import time
from pathlib import Path

from lit_gpt import GPT, Tokenizer
def check_model_device(model):
    return next(model.parameters()).device
# checkpoint_path = "/data/aniket/meta-llama/Llama-2-7b-chat-hf/lit_model.pth"
# checkpoint_dir = "/data/aniket/meta-llama/Llama-2-7b-chat-hf"
# model_name = "Llama-2-7b-chat-hf"
checkpoint_path = "lit-gpt/checkpoints/EleutherAI/pythia-1b/lit_model.pth"
checkpoint_dir = "lit-gpt/checkpoints/EleutherAI/pythia-1b/"
model_name = "pythia-1b"
checkpoints = torch.load(checkpoint_path)

fabric = L.Fabric(accelerator="cpu", precision="16", strategy="deepspeed_stage_3")
fabric.launch()

print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
t0 = time.time()
with fabric.init_module():
    model = GPT.from_name(model_name)
    model.load_state_dict(checkpoints)
print(f"Time to load model: {time.time() - t0:.02f} seconds.")
print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")