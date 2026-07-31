import os
import glob
import pandas as pd
import numpy as np
import json

print("======================================================")
print("  INITIATING NEURO-SYMBOLIC AUTO-IO-HMM PIPELINE")
print("======================================================\n")
# ==========================================
# PART 1: THE GLUE (Reading Dad's Scraper Data)
# ==========================================
def get_latest_harvested_data():
    print("[*] STEP 1: Scanning for harvested data...")
    
    # THE FIX: Added '**' and 'recursive=True' to search deep inside all subfolders
    list_of_files = glob.glob('data/**/*.csv', recursive=True) + glob.glob('data/**/*.parquet', recursive=True)
    
    if not list_of_files:
        print("[!] Still no data found. Check your folder structure.")
        return "Breaking: Sudden drought impacts regional crop yields, markets panic."
    
    # Get the newest file
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"[+] SUCCESS: Found real data file: {latest_file}")
    
    try:
        if latest_file.endswith('.parquet'):
            df = pd.read_parquet(latest_file)
        else:
            df = pd.read_csv(latest_file)
            
        raw_text = df.head(5).to_string()
        
        # PROOF: Print exactly what the script read from the file
        print("\n--- 🔎 PROOF: HERE IS THE REAL DATA I AM READING ---")
        print(raw_text)
        print("----------------------------------------------------\n")
        
        return raw_text
    except Exception as e:
        print(f"[!] Could not read file cleanly: {e}")
        return "Breaking: Unseasonal rainfall ruins harvest, subsidies announced."

# ==========================================
# PART 2: THE NEURAL BRIDGE (Mock LLM)
# ==========================================
def llm_sentiment_to_math(scraped_text):
    print("\n[*] STEP 2: Passing raw data to Neural Bridge (LLM)...")
    lower_text = scraped_text.lower()
    
    # The LLM contextualizes the messy data into a mathematical index
    if "drought" in lower_text or "crash" in lower_text or "panic" in lower_text:
        index = 0.15 # Catastrophic shock
        reason = "Detected severe negative market/weather conditions."
    elif "subsidy" in lower_text or "rainfall" in lower_text or "advisory" in lower_text:
        index = 0.95 # Positive shock
        reason = "Detected highly favorable government/market interventions."
    else:
        index = 0.60 # Status Quo
        reason = "Normal conditions detected in harvested data."

    print(f"[+] LLM Analysis Complete. Environmental Index: {index} ({reason})")
    return index

# ==========================================
# PART 3: THE AUTO-IO-HMM MATH ENGINE
# ==========================================
def apply_external_shock(base_matrix, env_index):
    dynamic_mat = np.copy(base_matrix)
    
    if env_index < 0.3:
        # Shift to Panic (State 3)
        dynamic_mat[:, 3] += 0.6 
    elif env_index >= 0.9:
        # Shift to Expansion (State 2)
        dynamic_mat[:, 2] += 0.5 
        
    row_sums = dynamic_mat.sum(axis=1, keepdims=True)
    return dynamic_mat / row_sums

def run_dynamic_hmm(env_index):
    print("\n[*] STEP 3: Ingesting covariates into Auto-IO-HMM...")
    
    states = [0, 1, 2, 3] 
    state_names = ["Survival", "Status Quo", "Expansion", "Panic (Discovered)"]
    start_probs = np.array([0.4, 0.4, 0.2, 0.0]) 

    base_trans_mat = np.array([
        [0.7, 0.2, 0.1, 0.0], 
        [0.3, 0.4, 0.3, 0.0], 
        [0.1, 0.3, 0.6, 0.0], 
        [0.4, 0.4, 0.2, 0.0]  
    ])
    
    emiss_mat = np.array([
        [0.8, 0.2, 0.0], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7], [0.9, 0.1, 0.0]
    ])

    # Simulate a timeline: 3 Normal months, then 1 Shock month based on the Scraper
    covariates = [0.6, 0.6, 0.6, env_index] 
    observed_actions = [1, 2, 1, 0] # Basics -> Tractor -> Basics -> Stops spending

    print("[*] Running Dynamic Viterbi Algorithm...")
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
            (prob, state) = max((V[t-1][y0] * current_trans_mat[y0][y] * emiss_mat[y][observed_actions[t]], y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath

    n = len(observed_actions) - 1
    (prob, state) = max((V[n][y], y) for y in states)
    
    print("\n======================================================")
    print("  FINAL INFERENCE RESULTS")
    print("======================================================")
    print(f"Observable Farmer Actions:  {observed_actions}")
    print(f"Environmental Index Array:  {covariates}")
    print("=> INFERRED LATENT STRATEGIES:")
    for time_step, strategy_id in enumerate(path[state]):
        print(f"   Month {time_step+1}: {state_names[strategy_id]}")
    print("======================================================\n")

# --- EXECUTE THE PIPELINE ---
raw_data = get_latest_harvested_data()
env_score = llm_sentiment_to_math(raw_data)
run_dynamic_hmm(env_score)