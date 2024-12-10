import os
import torch
from model import Model, ModelConfig
import pickle

class InferenceConfig():
    seed:int=0
    start:str="ROMEO:"
    temperature:float = 0.7
    max_new_tokens:int=250
    top_k:int=None

inference_config = InferenceConfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.manual_seed(inference_config.seed)

# Load the model and hyperparameters ｜
with open('model/model_config.pkl', 'rb') as f:
    model_config = pickle.load(f)

model = Model(model_config)
if model_config.compile:
    model = torch.compile(model)
model.load_state_dict(torch.load('model/model.ckpt', weights_only=True),strict=False)
model.eval()
model.to(device)

meta_path = os.path.join('data', 'Shakespeare', 'meta.pkl')
load_meta = os.path.exists(meta_path)
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

start_ids = encode(inference_config.start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation ｜
with torch.no_grad():
    y = model.generate(x, max_new_tokens=inference_config.max_new_tokens, temperature=inference_config.temperature, top_k=inference_config.top_k)
    print('---------------')
    print(decode(y[0].tolist()))
    print('---------------')

# Optionally, print model total of parameters ｜
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model param size: {total_params:,}")


