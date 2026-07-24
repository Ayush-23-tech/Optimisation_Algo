import numpy as np
import os
import warnings
from scipy.stats import wilcoxon
from opfunu.cec_based.cec2022 import * # =====================================================================
# --- 0. IEEE CEC COMPETITION STANDARDS & SETUP ---
# =====================================================================
DIM = 20
POP_SIZE = 50
MAX_FES = 1000000  # 1 Million FEs for 20D (IEEE Standard)
RUNS = 30
BOUND_MIN = -100.0
BOUND_MAX = 100.0

# Load official IEEE CEC 2022 seeds to guarantee 100% reproducible results
def load_official_seeds(filepath="Rand_Seeds.txt"):
    seeds = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                for p in parts:
                    if p:
                        # Convert IEEE format ('9.5800000e+02') -> 958.0 -> 958
                        seeds.append(int(float(p)))
        return seeds[:RUNS]
    except FileNotFoundError:
        print(f"[ERROR] {filepath} not found! Please ensure it is in the same folder.")
        return [202200 + i for i in range(RUNS)]

# =====================================================================
# --- 1. CLASSICAL RAO-1 ALGORITHM ---
# =====================================================================
def run_rao1(obj_func, dim, pop_size, max_fes, seed):
    np.random.seed(seed) 
    pop = np.random.uniform(BOUND_MIN, BOUND_MAX, (pop_size, dim))
    fitness = np.array([obj_func.evaluate(ind) for ind in pop])
    fes = pop_size
    
    while fes < max_fes:
        best_idx = np.argmin(fitness)
        worst_idx = np.argmax(fitness)
        X_best = pop[best_idx]
        X_worst = pop[worst_idx]
        
        for i in range(pop_size):
            if fes >= max_fes: break
            r1 = np.random.rand(dim)
            
            X_new = pop[i] + r1 * (X_best - X_worst)
            X_new = np.clip(X_new, BOUND_MIN, BOUND_MAX)
            
            f_new = obj_func.evaluate(X_new)
            fes += 1
            
            if f_new < fitness[i]:
                pop[i] = X_new
                fitness[i] = f_new
                
    return np.min(fitness)

# =====================================================================
# --- 2. HYBRID QUANTUM RAO (HQ-RAO) ---
# =====================================================================
def run_hq_rao(obj_func, dim, pop_size, max_fes, seed):
    np.random.seed(seed) 
    pop = np.random.uniform(BOUND_MIN, BOUND_MAX, (pop_size, dim))
    fitness = np.array([obj_func.evaluate(ind) for ind in pop])
    fes = pop_size
    
    while fes < max_fes:
        sorted_indices = np.argsort(fitness)
        best_idx = sorted_indices[0]
        worst_idx = sorted_indices[-1]
        X_best = pop[best_idx]
        X_worst = pop[worst_idx]
        
        ranks = np.zeros(pop_size)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank
            
        for i in range(pop_size):
            if fes >= max_fes: break
            H = ranks[i] / (pop_size - 1)
            
            # Classical Component
            r1 = np.random.rand(dim)
            X_classical = pop[i] + r1 * (X_best - X_worst)
            
            # Quantum Component (Gram-Schmidt)
            V_exp = X_best - X_worst
            norm_V = np.linalg.norm(V_exp)
            u = np.random.uniform(-1, 1, dim)
            
            if norm_V > 1e-12:
                u_perp = u - (np.dot(u, V_exp) / (norm_V**2)) * V_exp
            else:
                u_perp = u
                
            # Dirac Potential Well
            sigma = np.linalg.norm(pop[i] - X_worst) * np.log(1.0 / (np.random.rand() + 1e-15))
            X_quantum = X_best + sigma * u_perp
            
            # Hybridization
            X_new = H * X_classical + (1 - H) * X_quantum
            X_new = np.clip(X_new, BOUND_MIN, BOUND_MAX)
            
            f_new = obj_func.evaluate(X_new)
            fes += 1
            
            if f_new < fitness[i]:
                pop[i] = X_new
                fitness[i] = f_new
                
    return np.min(fitness)

# =====================================================================
# --- 3. MAIN BENCHMARK EXECUTION & STATS ---
# =====================================================================
if __name__ == "__main__":
    print("=" * 110)
    print(f" INITIATING IEEE CEC 2022 VALIDATION: RAO-1 vs HQ-RAO")
    print(f" Parameters: {MAX_FES} FEs | {RUNS} Runs | {DIM} Dimensions")
    print("=" * 110)
    print(f"{'Func':<5} | {'Rao-1 (Mean ± Std) [Median]':<35} | {'HQ-Rao (Mean ± Std) [Median]':<35} | {'W/T/L':<5} | {'p-val':<6}")
    print("-" * 110)
    
    competition_seeds = load_official_seeds("Rand_Seeds.txt")
    
    cec_functions = [
        F12022(ndim=DIM), F22022(ndim=DIM), F32022(ndim=DIM), F42022(ndim=DIM),
        F52022(ndim=DIM), F62022(ndim=DIM), F72022(ndim=DIM), F82022(ndim=DIM),
        F92022(ndim=DIM), F102022(ndim=DIM), F112022(ndim=DIM), F122022(ndim=DIM)
    ]
    
    # Global Stat Trackers
    total_wins = 0
    total_ties = 0
    total_losses = 0
    rao_friedman_sum = 0.0
    hq_friedman_sum = 0.0
    
    for f_idx, obj_func in enumerate(cec_functions, start=1):
        rao_errors = []
        hq_errors = []
        bias = f_idx * 100.0  
        
        for run in range(RUNS):
            seed = competition_seeds[run]
            
            rao_best = run_rao1(obj_func, DIM, POP_SIZE, MAX_FES, seed)
            hq_best = run_hq_rao(obj_func, DIM, POP_SIZE, MAX_FES, seed)
            
            # STRICT IEEE 1e-8 TRUNCATION RULE
            err_rao = rao_best - bias
            err_hq = hq_best - bias
            
            if err_rao < 1e-8: err_rao = 0.0
            if err_hq < 1e-8: err_hq = 0.0
            
            rao_errors.append(err_rao)
            hq_errors.append(err_hq)
            
        # Descriptive Statistics
        rao_mean, rao_std, rao_med = np.mean(rao_errors), np.std(rao_errors), np.median(rao_errors)
        hq_mean, hq_std, hq_med = np.mean(hq_errors), np.std(hq_errors), np.median(hq_errors)
        
        # Wilcoxon Signed-Rank Test for p-value
        if np.array_equal(rao_errors, hq_errors):
            p_val = 1.0  # Mathematically identical runs
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    stat, p_val = wilcoxon(rao_errors, hq_errors)
                except ValueError:
                    p_val = 1.0
                    
        # Determine Win/Tie/Loss based on 0.05 significance threshold
        if p_val < 0.05:
            if hq_mean < rao_mean:
                status = "Win"
                total_wins += 1
            else:
                status = "Loss"
                total_losses += 1
        else:
            status = "Tie"
            total_ties += 1
            
        # Calculate Friedman Rank based on mean performance
        if hq_mean < rao_mean:
            hq_friedman_sum += 1.0
            rao_friedman_sum += 2.0
        elif rao_mean < hq_mean:
            hq_friedman_sum += 2.0
            rao_friedman_sum += 1.0
        else:
            # Tie assigns 1.5 to both
            hq_friedman_sum += 1.5
            rao_friedman_sum += 1.5
            
        # Format strings for clean printing
        rao_str = f"{rao_mean:.2e} ± {rao_std:.2e} [{rao_med:.2e}]"
        hq_str = f"{hq_mean:.2e} ± {hq_std:.2e} [{hq_med:.2e}]"
        
        print(f"F{f_idx:<4} | {rao_str:<35} | {hq_str:<35} | {status:<5} | {p_val:.4f}")

    # Calculate final average Friedman Ranking
    total_funcs = len(cec_functions)
    avg_hq_rank = hq_friedman_sum / total_funcs
    avg_rao_rank = rao_friedman_sum / total_funcs

    print("=" * 110)
    print(" FINAL BENCHMARK TALLY: HQ-RAO vs RAO-1 (CEC 2022)")
    print("=" * 110)
    print(f" Wins - Ties - Losses      : {total_wins} - {total_ties} - {total_losses}")
    print(f" Total Functions Evaluated : {total_funcs}")
    print("-" * 110)
    print(" AVERAGE FRIEDMAN RANKING (Lower is Better)")
    print(f" 1. HQ-Rao                 : {avg_hq_rank:.2f}")
    print(f" 2. Classical Rao-1        : {avg_rao_rank:.2f}")
    print("=" * 110)