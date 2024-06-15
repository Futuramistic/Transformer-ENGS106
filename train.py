"""
Train a model
"""
import os
import torch
import tiktoken
from model import Model
import wandb

# Set GPU max allocation size to 512MB ｜
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.cuda.empty_cache()  # empty cache if necessary ｜

# Hyperparameters
h_params = {
    "d_model": 1024,  # Define our model dimension architecture ｜
    "batch_size": 4,  # How many batches per training step ｜
    "context_length": 16,  # Length of the token chunk each batch will receive ｜
    "num_blocks": 8,  # Number of transformer blocks ｜
    "num_heads": 4,  # Number of heads in Multi-head attention ｜
    "dropout": 0.1,  # Dropout rate ｜
    "max_iters": 2000,  # Total of training iterations ｜
    "learning_rate": 1e-3, # Learning rate ｜
    "eval_interval": 50,  # How often to evaluate the model ｜
    "eval_iters": 10,  # Number of iterations to average for evaluation ｜
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if it's available. ｜
    "epochs": 1,
    "TORCH_SEED": 1337
}
torch.manual_seed(h_params["TORCH_SEED"])

# WandB Tracking https://wandb.ai/
# run = wandb.init(
#     project="LLMZhang_lesson_2",
#     # Track hyperparameters and run metadata
#     config={
#         "d_model": h_params["d_model"],
#         "batch_size": h_params["batch_size"],
#         "context_length": h_params["context_length"],
#         "max_iters": h_params["max_iters"],
#         "learning_rate": h_params["learning_rate"],
#         "epochs": h_params["epochs"],
#     },
# )


# Prepare Datasets ｜
with open('data/hardware_prices.csv', 'r', encoding="utf-8") as file:
    text = file.read()

# Using TikToken (Same as GPT3) as tokenizer ｜
tokenizer = tiktoken.get_encoding("cl100k_base")
tokenized_text = tokenizer.encode(text)
max_token_value = max(tokenized_text)+1  # the maximum value of the tokenized numbers
h_params['max_token_value'] = max_token_value # push max_token_value to hyperparameters for model initialization
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=h_params['device'])

total_tokens = tokenizer.encode_ordinary(text)
print(f"Total: {len(total_tokens):,} tokens")


# Split train and validation data ｜
train_size = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_size]
val_data = tokenized_text[train_size:]

# Initialize the model ｜
model = Model(h_params).to(h_params['device'])

# WandB LogMagic ｜ WandB
# wandb.watch(model, log_freq=100)

# get batch data ｜
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - h_params['context_length'], size=(h_params['batch_size'],))
    x = torch.stack([data[idx:idx + h_params['context_length']] for idx in idxs]).to(h_params['device'])
    y = torch.stack([data[idx + 1:idx + h_params['context_length'] + 1] for idx in idxs]).to(h_params['device'])
    return x, y

# calculate the loss ｜
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(h_params['eval_iters'])
        for k in range(h_params['eval_iters']):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Create the optimizer ｜
optimizer = torch.optim.AdamW(model.parameters(), lr=h_params['learning_rate'])
for step in range(h_params['max_iters']):
    if step % h_params['eval_interval'] == 0 or step == h_params['max_iters'] - 1:
        losses = estimate_loss()
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:', round(losses['valid'].item(), 3))

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    # Logging Trace ｜
    # wandb.log({"train loss": round(losses['train'].item(), 3)})  # WandB validation loss tracking
    # wandb.log({"valid loss": round(losses['valid'].item(), 3)})  # WandB validation loss tracking
    # Backpropagation ｜
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model ｜
torch.save({
    'model_state_dict': model.state_dict(),
    'h_params': h_params
}, 'model/model.ckpt')
