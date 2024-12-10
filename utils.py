import torch
import tqdm
from torch.nn import functional as F
from dataset import getBatch
import pickle

def loss_func(logits,targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

# calculate the loss ｜
@torch.no_grad()
def validation(model,train_config):
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(train_config.eval_iters)
        for k in range(train_config.eval_iters):
            x_batch, targets = getBatch(split, train_config.train_dataloader, train_config.test_dataloader)
            with train_config.ctx:
                logits = model(x_batch)
                loss = loss_func(logits,targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(model, train_config):
    for step in tqdm.tqdm(range(train_config.max_iters)):
        if step % train_config.eval_interval == 0 or step == train_config.max_iters - 1:
            losses = validation(model,train_config)
            tqdm.tqdm.write(f'Step: {step} Training Loss: {round(losses['train'].item(), 3)} Validation Loss: {round(losses['valid'].item(), 3)}')
        xb, targets = getBatch('train', train_config.train_dataloader, train_config.test_dataloader)
        with train_config.ctx:
            logits = model(xb)
            loss = loss_func(logits,targets)
        train_config.scaler.scale(loss).backward()
        if train_config.grad_clip != 0.0:
            train_config.scaler.unscale_(train_config.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        train_config.scaler.step(train_config.optimizer)
        train_config.scaler.update()
        train_config.optimizer.zero_grad(set_to_none=True)

def inference(model, inference_config):
    with open(inference_config.meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    start_ids = encode(inference_config.start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=inference_config.device)[None, ...])

    # run generation ｜
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=inference_config.max_new_tokens, temperature=inference_config.temperature, top_k=inference_config.top_k)
        return decode(y[0].tolist())