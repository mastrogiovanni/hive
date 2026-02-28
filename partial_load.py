import torch
from transformers import AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file
import json

home = "/home/michele/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"

with open(f"{home}/model.safetensors.index.json") as f:
    index = json.load(f)
    
print(index)

model_id = "meta-llama/Meta-Llama-3-8B"
config = AutoConfig.from_pretrained(model_id)

# Create an empty shell of the model
with torch.device("cuda"):
    model = AutoModelForCausalLM.from_config(config)
    
# Paths to the local files you downloaded
path_to_weights = f"{home}/model-00003-of-00004.safetensors"
state_dict = load_file(path_to_weights)

for key in state_dict.keys():
    print(key)  # This will show you the layer names and help you identify the split point

# Filter the state dict for only your target layers
# Note: Layer naming varies (e.g., 'model.layers.23...')
subset_dict = {k: v for k, v in state_dict.items() if any(f".{i}." in k for i in range(23, 57))}

# Move the subset to your actual device (CPU/GPU)
model.load_state_dict(subset_dict, strict=False)