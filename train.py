"""
Train a model
"""
import os
import torch
from model import Model, ModelConfig
from dataset import getData, getVocabSize
import pickle
from contextlib import nullcontext
from utils import train

class TrainConfig:

    # Parameters to modify:
    batch_size: int = 64  # How many batches per training step
    max_iters: int = 2000  # Total of training iterations
    learning_rate: float=1e-3 # Learning rate
    grad_clip: float=1.0 # Maximium magnitude of gradient
    eval_interval: int=50 # How often to evaluate the model
    eval_iters: int=10 # Number of iterations to average for evaluation
    seed: int=1337 # Random seed (can change the results)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # These are responsible for correct training given GPU (DO NOT MODIFY)
    dtype: str =  'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
    scaler = torch.amp.GradScaler(device,enabled=(dtype == 'float16'))
    
    # Populated by the script (DO NOT MODIFY)
    train_dataloader: None
    test_dataloader: None
    optimizer: None
    

model_config = ModelConfig()
train_config = TrainConfig()

torch.manual_seed(train_config.seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# Load data
data_dir = os.path.join('data', 'Shakespeare')
model_config.vocab_size = getVocabSize(data_dir)
train_config.train_dataloader, train_config.test_dataloader = getData(data_dir,model_config,train_config)

# Initialize the model
model = Model(model_config).to(train_config.device)
if model_config.compile:
    model = torch.compile(model)

# Create the optimizer and train
train_config.optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
train(model,train_config)

# Save the model

torch.save(model.state_dict(), 'model/model.ckpt')
with open('model/model_config.pkl','wb') as f:
    pickle.dump(model_config, f)