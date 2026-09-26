import os
import glob
import pandas as pd
import numpy as np

def get_latest_harvested_data(data_dir='data'):
    """
    Scans the data directory recursively for the latest extracted file.
    Returns a string sample of the data for LLM context ingestion.
    """
    files = glob.glob(f'{data_dir}/**/*.csv', recursive=True) + \
            glob.glob(f'{data_dir}/**/*.parquet', recursive=True)
    
    if not files:
        return None
    
    latest_file = max(files, key=os.path.getctime)
    
    try:
        if latest_file.endswith('.parquet'):
            return pd.read_parquet(latest_file).head(5).to_string()
        return pd.read_csv(latest_file).head(5).to_string()
    except Exception:
        return None

def extract_covariates(scraped_text):
    """
    Evaluates unstructured text to derive a continuous environmental index.
    """
    if not scraped_text:
        return 0.60 
        
    lower_text = scraped_text.lower()
    if any(word in lower_text for word in ["drought", "crash", "panic", "crisis"]):
        return 0.15 
    elif any(word in lower_text for word in ["subsidy", "rainfall", "advisory", "boom"]):
        return 0.95 
        
    return 0.60 

def apply_external_shock(base_matrix, env_index):
    """
    Dynamically mutates transition probabilities using a Multinomial Logit 
    (Softmax) function conditioned on the real-time covariate (env_index).
    """
    K = base_matrix.shape[0]
    dynamic_mat = np.zeros_like(base_matrix)
    
    W = np.array([
        [-1.0,  0.5,  2.0, -2.5],  
        [-0.5,  1.0,  1.5, -2.0],  
        [ 0.0,  0.5,  2.0, -1.0],  
        [-2.0, -1.0,  0.0,  3.0]   
    ])
                  
    b = np.log(base_matrix + 1e-9) 

    for i in range(K):
        logits = (W[i] * env_index) + b[i]
        dynamic_mat[i, :] = np.exp(logits) / np.sum(np.exp(logits))
        
    return dynamic_mat

def run_dynamic_viterbi(env_index):
    """
    Executes a covariate-conditioned Viterbi decoding algorithm.
    """
    states = [0, 1, 2, 3] 
    state_names = ["Survival", "Status Quo", "Expansion", "Panic"]
    start_probs = np.array([0.4, 0.4, 0.2, 0.0]) 

    base_trans_mat = np.array([
        [0.7, 0.2, 0.1, 0.0], 
        [0.3, 0.4, 0.3, 0.0], 
        [0.1, 0.3, 0.6, 0.0], 
        [0.4, 0.4, 0.2, 0.0]  
    ])
    
    emiss_mat = np.array([
        [0.8, 0.2, 0.0], 
        [0.3, 0.6, 0.1], 
        [0.1, 0.2, 0.7], 
        [0.9, 0.1, 0.0]
    ])

    covariates = [0.6, 0.6, 0.6, env_index] 
    observed_actions = [1, 2, 1, 0] 

    V = [{}]
    path = {}
    
    for y in states:
        V[0][y] = start_probs[y] * emiss_mat[y][observed_actions[0]]
        path[y] = [y]

    for t in range(1, len(observed_actions)):
        V.append({})
        newpath = {}
        current_trans_mat = apply_external_shock(base_trans_mat, covariates[t])

        for y in states:
            prob, state = max(
                (V[t-1][y0] * current_trans_mat[y0][y] * emiss_mat[y][observed_actions[t]], y0) 
                for y0 in states
            )
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath

    n = len(observed_actions) - 1
    prob, state = max((V[n][y], y) for y in states)
    
    return observed_actions, covariates, [state_names[idx] for idx in path[state]]

if __name__ == "__main__":
    raw_data = get_latest_harvested_data()
    env_score = extract_covariates(raw_data)
    actions, environment, strategies = run_dynamic_viterbi(env_score)
    
    print(f"Observable Actions: {actions}")
    print(f"Environmental Covariates: {environment}")
    print(f"Inferred Strategies: {strategies}")
