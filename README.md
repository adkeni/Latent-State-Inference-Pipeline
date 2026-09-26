# Latent-State-Inference-Pipeline (LSIP)
**A Neuro-Symbolic Auto-IO-HMM Architecture for Imputing Behavioral Data**

---

## Overview
This repository contains a reproducible methods framework for imputing missing qualitative behavioral data (**Latent States**) from quantitative macro-environmental data and end-point actions (**Observed States**). 

As detailed in our accompanying journal submission, this pipeline serves as a synthetic proof-of-concept. It demonstrates how a decoupled architecture can extract external context from public text and propagate it into a probabilistic sequence model, establishing an auditable baseline for data-sparse environments.

### The Breakthrough: Auto-IO-HMM
Traditional Hidden Markov Models assume temporal stationarity. To relax this restriction, this project introduces a covariate-conditioned Auto-IO-HMM. It utilizes a neuro-symbolic extraction layer (LLM) to convert unstructured text into a contextual covariate vector ($x_t$). This covariate dynamically mutates the HMM's transition probabilities at time $t$ using a multinomial-logit transition function, allowing the model to adapt to exogenous systemic shocks while preserving standard Viterbi decoding.

---

## Decoupled Architecture
To ensure scalability and prevent data ingestion failures from corrupting the mathematical engine, this project utilizes a strict decoupled architecture:

1. **The Eyes (Data Harvester):** A custom data-harvesting architecture—adapted from a baseline prototype—that autonomously extracts unstructured regional data and time-series metrics from public APIs.
2. **The Neural Bridge (NLP Extraction):** An extraction layer that uses Natural Language Processing to contextualize raw text into a continuous mathematical "Environmental Index" ($X_t$).
3. **The Brain (Auto-IO-HMM Engine):** A dynamic inference engine built from scratch in NumPy utilizing a modified Viterbi algorithm. 

---

## Mathematical Formulation
Instead of a static transition matrix $A$, the Auto-IO-HMM transition probabilities mutate at every time step $t$ using a Softmax function conditioned on the real-time external shock $X_t$:

$$P(H_t = j \mid H_{t-1} = i, X_t) = \frac{\exp(W_{ij} X_t + b_{ij})}{\sum_{k=1}^K \exp(W_{ik} X_t + b_{ik})}$$

Where $W$ represents the learned weights and $b$ represents the biases optimized during the Baum-Welch training phase, ensuring the model accurately diagnoses systemic shifts in human strategy.

---

## Repository Structure
```text
Latent-State-Inference-Pipeline/
│
├── auto_io_hmm.py           # The core dynamic mathematical inference engine
├── mal_data_harvester.py    # Automated web-scraping and data ingestion module
├── requirements.txt         # Environment dependencies
└── README.md                # Project documentation
