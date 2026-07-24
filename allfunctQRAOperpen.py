import numpy as np
import time
import warnings
import opfunu.cec_based.cec2014 as cec2014
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURATION
# ======================================================
POP_SIZE = 50        
DIMENSIONS = 30      
MAX_FES = 30000     # Low FE budget for quick diagnostic verification (Scale to 300,000 for final paper)
RUNS = 10            

LOWER_BOUND = -100.0
UPPER_BOUND = 100.0

def initialize_population(pop_size, dimensions):
    return np.random.uniform(LOWER_BOUND, UPPER_BOUND, (pop_size, dimensions))

# ======================================================
# 1. CLASSICAL RAO-1
# ======================================================
def rao1_fe_capped(objective_function):
    population = initialize_population(POP_SIZE, DIMENSIONS)
    fes = POP_SIZE 
    fitness = np.array([objective_function(np.asarray(x, dtype=np.float64)) for x in population])
    
    while fes < MAX_FES:
        best = population[np.argmin(fitness)]
        worst = population[np.argmax(fitness)]
        new_population = []

        for i, x in enumerate(population):
            if fes >= MAX_FES: 
                new_population.append(x)
                continue
            
            r = np.random.rand(DIMENSIONS)
            x_new = np.clip(x + r * (best - worst), LOWER_BOUND, UPPER_BOUND)
            new_fit = objective_function(np.asarray(x_new, dtype=np.float64))
            fes += 1

            if new_fit < fitness[i]:
                new_population.append(x_new)
                fitness[i] = new_fit
            else:
                new_population.append(x)
                
        population = np.array(new_population)
    return np.min(fitness)

# ======================================================
# 2. RANK-H ONLY
# ======================================================
def rank_h_only_fe_capped(objective_function):
    population = initialize_population(POP_SIZE, DIMENSIONS)
    fes = POP_SIZE
    fitness = np.array([objective_function(np.asarray(x, dtype=np.float64)) for x in population])
    sigma_min = (UPPER_BOUND - LOWER_BOUND) / np.sqrt(POP_SIZE)

    while fes < MAX_FES:
        best = population[np.argmin(fitness)]
        worst = population[np.argmax(fitness)]
        
        ranks = np.argsort(np.argsort(fitness))
        H = ranks / (POP_SIZE - 1.0) 
        v_exp = best - worst
        new_population = []

        for i, x in enumerate(population):
            if fes >= MAX_FES: 
                new_population.append(x)
                continue
            
            h = H[i] 
            x_classical = x + np.random.rand(DIMENSIONS) * v_exp
            
            u = np.random.randn(DIMENSIONS)
            u_dir = u / (np.linalg.norm(u) + 1e-14)
            
            sigma = max(np.linalg.norm(x - worst), sigma_min) * np.log(1.0 / (np.random.rand() + 1e-14))
            x_quantum = best + (np.random.choice([-1, 1]) * sigma * u_dir)
            
            x_new = (h * x_classical) + ((1.0 - h) * x_quantum)
            x_new = np.clip(x_new, LOWER_BOUND, UPPER_BOUND)
            new_fit = objective_function(np.asarray(x_new, dtype=np.float64))
            fes += 1

            if new_fit < fitness[i]:
                new_population.append(x_new)
                fitness[i] = new_fit
            else:
                new_population.append(x)
                
        population = np.array(new_population)
    return np.min(fitness)

# ======================================================
# 3. RANK-H + ORTHOGONAL TUNNELING
# ======================================================
def rank_h_orthogonal_fe_capped(objective_function):
    population = initialize_population(POP_SIZE, DIMENSIONS)
    fes = POP_SIZE
    fitness = np.array([objective_function(np.asarray(x, dtype=np.float64)) for x in population])
    sigma_min = (UPPER_BOUND - LOWER_BOUND) / np.sqrt(POP_SIZE)

    while fes < MAX_FES:
        best = population[np.argmin(fitness)]
        worst = population[np.argmax(fitness)]
        
        ranks = np.argsort(np.argsort(fitness))
        H = ranks / (POP_SIZE - 1.0) 
        v_exp = best - worst
        v_dir = v_exp / (np.linalg.norm(v_exp) + 1e-14)
        new_population = []

        for i, x in enumerate(population):
            if fes >= MAX_FES: 
                new_population.append(x)
                continue
            
            h = H[i] 
            x_classical = x + np.random.rand(DIMENSIONS) * v_exp
            
            u = np.random.randn(DIMENSIONS)
            u_perp = u - np.dot(u, v_dir) * v_dir
            u_perp = u_perp / (np.linalg.norm(u_perp) + 1e-14)
            
            sigma = max(np.linalg.norm(x - worst), sigma_min) * np.log(1.0 / (np.random.rand() + 1e-14))
            x_quantum = best + (np.random.choice([-1, 1]) * sigma * u_perp)
            
            x_new = (h * x_classical) + ((1.0 - h) * x_quantum)
            x_new = np.clip(x_new, LOWER_BOUND, UPPER_BOUND)
            new_fit = objective_function(np.asarray(x_new, dtype=np.float64))
            fes += 1

            if new_fit < fitness[i]:
                new_population.append(x_new)
                fitness[i] = new_fit
            else:
                new_population.append(x)
                
        population = np.array(new_population)
    return np.min(fitness)

# ======================================================
# EXPERIMENT EXECUTION ENGINE
# ======================================================
target_functions = [1,2,3, 4, 5,6,7,8,9, 10, 11,12,14,15,16,17,18,19,20, 21, 23]
benchmark_suite = []

for f_num in target_functions:
    class_name = f"F{f_num}2014"
    if hasattr(cec2014, class_name):
        cec_class = getattr(cec2014, class_name)
        bias = f_num * 100.0
        benchmark_suite.append((cec_class(ndim=DIMENSIONS).evaluate, f"F{f_num}", bias))

print(f"STRICT EVALUATION: {MAX_FES} FEs | {RUNS} Runs | {DIMENSIONS}D")
print("="*110)

def compute_wilcoxon(baseline, variant):
    # If arrays are exactly identical, Wilcoxon cannot compute a p-value
    if np.array_equal(baseline, variant):
        return "N/A"
    try:
        _, p_val = wilcoxon(baseline, variant)
        return f"{p_val:.4f}"
    except ValueError:
        # Handles cases where all differences are zero
        return "N/A"

for obj_func, name, bias in benchmark_suite:
    print(f"\nEvaluating Landscape: {name}")
    print("-" * 110)
    
    err_rao, err_rank, err_ortho = [], [], []

    for run in range(RUNS):
        err_rao.append(rao1_fe_capped(obj_func) - bias)
        err_rank.append(rank_h_only_fe_capped(obj_func) - bias)
        err_ortho.append(rank_h_orthogonal_fe_capped(obj_func) - bias)

    # Compute Statistics
    m_rao, s_rao = np.mean(err_rao), np.std(err_rao)
    m_rank, s_rank = np.mean(err_rank), np.std(err_rank)
    m_ortho, s_ortho = np.mean(err_ortho), np.std(err_ortho)
    
    p_rank = compute_wilcoxon(err_rao, err_rank)
    p_ortho = compute_wilcoxon(err_rao, err_ortho)

    # Print clean formatted markdown rows
    print(f"{'Algorithm':<25} | {'Mean Error':<18} | {'Std Dev':<18} | {'Wilcoxon p-val (vs Rao)'}")
    print(f"{'-'*25}-|-{'-'*18}-|-{'-'*18}-|-{'-'*24}")
    print(f"{'1. Classical Rao':<25} | {m_rao:<18.4e} | {s_rao:<18.4e} | {'Baseline':<24}")
    print(f"{'2. Rank-H Only':<25} | {m_rank:<18.4e} | {s_rank:<18.4e} | {p_rank:<24}")
    print(f"{'3. Rank-H + Ortho':<25} | {m_ortho:<18.4e} | {s_ortho:<18.4e} | {p_ortho:<24}")