import os
import torch
from model import Model
import pickle
from utils import inference

class InferenceConfig():
    seed:int=0 # Random seed (impacts the output)
    start:str="ROMEO:" # Starting prompt to generate from
    temperature:float = 0.7 # Degree of 'creativity': 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    max_new_tokens:int=250 # Length of the generated sequence in tokens
    top_k:int=None  # Retain only the top k most likely tokens, clamp others to have 0 probability (None - no clamp)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

inference_config = InferenceConfig()
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
model.to(inference_config.device)

inference_config.meta_path = os.path.join('data', 'Shakespeare', 'meta.pkl')

print('GENERATED TEXT:')
print('-----------------')
print(inference(model, inference_config))
print('-----------------')


# Optionally, print model total of parameters ｜
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model param size: {total_params:,}")


