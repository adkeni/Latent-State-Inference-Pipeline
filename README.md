# Latent-State-Inference-Pipeline (LSIP)
**A Neuro-Symbolic Auto-IO-HMM Architecture for Imputing Behavioral Data**

---

## Overview
This repository contains a domain-agnostic data imputation pipeline designed to infer missing qualitative human behavioral data (**Hidden States**) from quantitative macro-environmental data and end-point actions (**Observed States**). 

While initially deployed as a Proof of Concept (PoC) in agricultural economics to infer unrecorded farmer strategies based on public weather and market data, the pipeline is architected to be highly scalable across any data-scarce industry (e.g., finance, healthcare, supply chain logistics).

### The Breakthrough: Auto-IO-HMM
This project introduces the **Autonomous Input-Output Hidden Markov Model (Auto-IO-HMM)**. Unlike standard HMMs which rely on rigid, static transition matrices, this architecture dynamically mutates transition probabilities in real-time. It utilizes exogenous covariates (e.g., sudden market crashes or climate events) extracted via a neuro-symbolic Large Language Model (LLM) bridge to recalculate behavioral probabilities at time $t$.

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
