# Transformer
This demo showcases training a **Transformer** based **Large Language Model (LLM)**. 

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT), it is designed to be simple and easy to understand, making it an excellent starting point for beginners learning how to train an LLM from scratch using PyTorch.

The demo is trained on a 11.7 kB [hardware_prices](https://huggingface.co/datasets/Astrale0031/hardware_prices/raw/main/hardware_prices.csv) dataset, nd the model size is about 1.10 GB. I trained on a single NVIDIA GeForce RTX 4090 GPU, and the training time takes about 2 hours, result in approximately 296,291,530 parameters.


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

2. Run train.py

 The model will start training on the dataset. Training & validation `losses` will be printed on the console screen, something like:

```bash
Step: 0 Training Loss: 11.594 Validation Loss: 11.575
Step: 50 Training Loss: 2.617 Validation Loss: 2.116
Step: 100 Training Loss: 2.137 Validation Loss: 2.078
Step: 150 Training Loss: 2.227 Validation Loss: 2.081
Step: 200 Training Loss: 2.076 Validation Loss: 1.897
Step: 250 Training Loss: 1.682 Validation Loss: 1.844
...
```

The training loss will decrease as the training goes on.  The model will be saved under name `model-ckpt.pt`. 

Feel free to change some of the hyperparameters on the top of the `train.py` file, and see how it affects the training process.

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
