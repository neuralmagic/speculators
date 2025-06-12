from safetensors.torch import  load_file

filename = "Eagle3take2B/state_5/model.safetensors" # Replace your_file.safetensors with the actual file name
state_dict={}

f=load_file(filename)
keys = list(f.keys())
print("Keys in the safetensors file:", keys)
for key, tensor in f.items():
    state_dict[key]=tensor

state_dict['layers.0.hidden_norm.weight']=state_dict['hidden_norm.weight']
del state_dict['hidden_norm.weight']
state_dict['layers.0.input_layernorm.weight']=state_dict['input_layernorm.weight']
del state_dict['input_layernorm.weight']
state_dict['norm.weight']=state_dict["lm_head_layernorm.weight"]
del state_dict["lm_head_layernorm.weight"]



filename = "Eagle3take2B/state_5/model_1.safetensors" # Replace your_file.safetensors with the actual file name
f=load_file(filename)
for key, tensor in f.items():
    state_dict["lm_head."+key]=tensor


import torch
from safetensors.torch import save_file


save_file(state_dict, "eagle3/model.safetensors")

print(state_dict.keys())