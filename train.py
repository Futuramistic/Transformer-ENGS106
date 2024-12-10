"""
Train a model
"""
import os
import torch
from model import Model, ModelConfig
import tqdm
from torch.nn import functional as F
from dataset import TokenDataset
from torch.utils.data import DataLoader
import numpy as np
import pickle
from contextlib import nullcontext

class TrainConfig:
    batch_size: int = 64  # How many batches per training step ｜
    max_iters: int = 2000  # Total of training iterations ｜
    learning_rate: float=1e-3 # Learning rate ｜
    grad_clip: float=1.0
    eval_interval: int=50  # How often to evaluate the model ｜
    eval_iters: int=10 # Number of iterations to average for evaluation ｜
    seed: int=1337

model_config = ModelConfig()
train_config = TrainConfig()

dtype =  'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(train_config.seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

data_dir = os.path.join('data', 'Shakespeare')
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
model_config.vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304

test_data = np.memmap(os.path.join(data_dir,'val.bin'), dtype=np.uint16, mode='r')
train_data = np.memmap(os.path.join(data_dir,'train.bin'), dtype=np.uint16, mode='r')
train_dataset = TokenDataset(train_data,model_config.context_length,device)
test_dataset = TokenDataset(test_data,model_config.context_length,device)
train_dataloader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=train_config.batch_size, shuffle=True)

# Initialize the model ｜
model = Model(model_config).to(device)
if model_config.compile:
    model = torch.compile(model)

# get batch data ｜
def get_batch(split: str):
    data = train_dataloader if split == 'train' else test_dataloader
    x,y = next(iter(data))
    return x, y

# calculate the loss ｜
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(train_config.eval_iters)
        for k in range(train_config.eval_iters):
            x_batch, targets = get_batch(split)
            with ctx:
                logits = model(x_batch)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Create the optimizer ｜
optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
scaler = torch.amp.GradScaler('cuda',enabled=(dtype == 'float16'))

for step in tqdm.tqdm(range(train_config.max_iters)):
    if step % train_config.eval_interval == 0 or step == train_config.max_iters - 1:
        losses = estimate_loss()
        tqdm.tqdm.write(f'Step: {step} Training Loss: {round(losses['train'].item(), 3)} Validation Loss: {round(losses['valid'].item(), 3)}')
    xb, targets = get_batch('train')
    with ctx:
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    scaler.scale(loss).backward()
    if train_config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

# Save the model ｜
torch.save(model.state_dict(), 'model/model.ckpt')
with open('model/model_config.pkl','wb') as f:
    pickle.dump(model_config, f)