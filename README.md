# Transformers-GPT

[![Author](https://img.shields.io/badge/Author-harshu0117-blue)](https://github.com/harshu0117) [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=flat)](https://pytorch.org/) [![OpenAI](https://img.shields.io/badge/OpenAI-4A4A55?logo=openai&logoColor=white&style=flat)](https://arxiv.org/abs/2005.14165) [![Transformers](https://img.shields.io/badge/Transformers-00ADEF?logo=transformers&logoColor=white&style=flat)](https://arxiv.org/abs/1706.03762)



A PyTorch implementation of the Transformer architecture, inspired by the seminal paper [Attention is All You Need](https://arxiv.org/abs/1706.03762), and OpenAI's GPT models. This repository contains code for building, training, and evaluating transformer-based language models from scratch.

---

## Table of Contents
- [Overview](#overview)
- [Files](#files)
- [Architecture](#architecture)
- [References & Resources](#references--resources)
- [Credits](#credits)

---

## Overview

This project demonstrates a transformer-based language model, closely following the original transformer design and the decoder-only architecture used in OpenAI's GPT-2 and GPT-3. The code is modular and educational, making it easy to understand the core components of modern large language models (LLMs).

---

## Files

- **transformer.py**: Implements the full transformer architecture in PyTorch, including:
  - Input Embeddings
  - Positional Encoding
  - Multi-Head Attention
  - Feed Forward Networks
  - Layer Normalization
  - Encoder and Decoder blocks
  - Residual Connections
  - Projection Layer
  - Model builder function

- **gpt124_openai.ipynb**: Jupyter notebook for experimenting with the transformer model, including:
  - Loading and preprocessing data
  - Training and evaluation loops
  - Example usage of the transformer for text generation

---

## Architecture

The transformer model is based on the following key components:

- **Input Embeddings**: Converts token indices to dense vectors.
- **Positional Encoding**: Adds information about token positions to embeddings.
- **Multi-Head Attention**: Allows the model to focus on different parts of the input sequence.
- **Feed Forward Networks**: Applies non-linear transformations to each position.
- **Residual Connections & Layer Normalization**: Stabilize and speed up training.
- **Encoder & Decoder**: The original transformer uses both; GPT models use only the decoder stack for autoregressive text generation.

Below is a standard diagram of the transformer architecture (from the original paper):

![Transformer Architecture](https://towardsdatascience.com/wp-content/uploads/2020/11/1ZCFSvkKtppgew3cc7BIaug.png)

---

## References & Resources

### Papers
- <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention is All You Need (Vaswani et al., 2017)</a>
- <a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" target="_blank">OpenAI GPT-2 (Radford et al., 2019)</a>
- <a href="https://arxiv.org/abs/2005.14165" target="_blank">OpenAI GPT-3 (Brown et al., 2020)</a>

### Transformer Architecture Visualizations
- <a href="https://jalammar.github.io/illustrated-transformer/" target="_blank">The Illustrated Transformer by Jay Alammar</a>

### YouTube Reference
- <a href="https://www.youtube.com/@vizuara" target="_blank">Vizuara AI - YouTube Channel</a>



## Credits

- Implementation inspired by the original transformer paper and OpenAI's GPT models.
- For more visual explanations, check out [Vizuara AI](https://www.youtube.com/@vizuara) on YouTube. 
