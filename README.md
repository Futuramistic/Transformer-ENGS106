# Transformer
This demo showcases training a **Transformer** based **Large Language Model (LLM)**. 
Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT), it is designed to be simple and easy to understand, making it an excellent starting point for beginners learning how to train an LLM from scratch using PyTorch.

## Install and start

```bash
pip install -U torch numpy tiktoken
```

wandb is used to record the training process(optional)：

```bash
pip install wandb
```

wandb (Weights & Bias) is the most versatile library for model training monitoring. It can help you record the metrics, hyperparameters, model structure, model files, etc. during model training. You can register an account on [official website](https://wandb.ai/), and then add the following code to your code to start recording the training process.

## Catalogs

- `data/`: Houses the sample dataset used for training
- `model/`: the model that will be trained
- model.py: Transformer model logic code
- train.py: training code
- inference.py: inference code

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)The original paper of Transformer architecture.
+ [nanoGPT](https://github.com/karpathy/nanoGPT)Andrej Karpathy's famous video tutorial on how to build a GPT model from scratch.
* [Transformers from Scratch](https://blog.matdmiller.com/posts/2023-06-10_transformers/notebook.html)A clear and easy implementation of Andrej's video contents by Mat Miller.
