# Transformer
This demo showcases training a **Transformer** based **Large Language Model (LLM)**. 

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT), it is designed to be simple and easy to understand, making it an excellent starting point for beginners learning how to train an LLM from scratch using PyTorch.

The demo is trained on

## Install and start
1. Install dependencies
```bash
pip install -U torch numpy tiktoken
```

wandb is used to record the training process(optional)：

```bash
pip install wandb
```

wandb (Weights & Bias) is the most versatile library for model training monitoring. It can help you record the metrics, hyperparameters, model structure, model files, etc. during model training. You can register an account on [official website](https://wandb.ai/), and then add the following code to your code to start recording the training process.

2. Run model.py
 The model will start training on the dataset. Training & validation `losses` will be printed on the console screen, something like:

```bash
Step: 0 Training Loss: 11.68 Validation Loss: 11.681
Step: 20 Training Loss: 10.322 Validation Loss: 10.287
Step: 40 Training Loss: 8.689 Validation Loss: 8.783
Step: 60 Training Loss: 7.198 Validation Loss: 7.617
Step: 80 Training Loss: 6.795 Validation Loss: 7.353
Step: 100 Training Loss: 6.598 Validation Loss: 6.789
...
```

The training loss will decrease as the training goes on.  The model will be saved under name `model-ckpt.pt`. 

Feel free to change some of the hyperparameters on the top of the `model.py` file, and see how it affects the training process.

## Catalogs

- `data/`: Houses the sample dataset used for training
- `model/`: the model that will be trained
- model.py: Transformer model logic code
- train.py: training code
- inference.py: inference code

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) The original paper of Transformer architecture.
+ [nanoGPT](https://github.com/karpathy/nanoGPT) Andrej Karpathy's famous video tutorial on how to build a GPT model from scratch.
* [Transformers from Scratch](https://blog.matdmiller.com/posts/2023-06-10_transformers/notebook.html) A clear and easy implementation of Andrej's video contents by Mat Miller.
