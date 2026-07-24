import os
import warnings

# ======================================================
# ABSOLUTE SILENCE PROTOCOL (No Warnings Allowed)
# ======================================================
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import opfunu.cec_based.cec2014 as cec2014
from scipy.stats import wilcoxon, rankdata
import concurrent.futures
import multiprocessing

# ======================================================
# THE TOGGLE SWITCH (Change this for Run 2)
# ======================================================
TARGET_ALGORITHM = "RAO3"  # Change to "RAO3" when ready

# ======================================================
# EXPERIMENT CONFIGURATION (IEEE Standard)
# ======================================================
POP_SIZE = 50        
DIMENSIONS = 30      
MAX_FES = 30000     
RUNS = 30            
LOWER_BOUND = -100.0
UPPER_BOUND = 100.0

def initialize_population(pop_size, dimensions):
    return np.random.uniform(LOWER_BOUND, UPPER_BOUND, (pop_size, dimensions))

# ======================================================
# ALGORITHM DEFINITIONS
# ======================================================
def rao2_fe_capped(obj_func):
    pop = initialize_population(POP_SIZE, DIMENSIONS)
    fes = POP_SIZE 
    fit = np.array([obj_func(np.asarray(x, dtype=np.float64)) for x in pop])
    while fes < MAX_FES:
        best, worst = pop[np.argmin(fit)], pop[np.argmax(fit)]
        new_pop = []
        for i, x in enumerate(pop):
            if fes >= MAX_FES: 
                new_pop.append(x)
                continue
            r1, r2 = np.random.rand(DIMENSIONS), np.random.rand(DIMENSIONS)
            l = np.random.randint(0, POP_SIZE)
            while l == i: l = np.random.randint(0, POP_SIZE)
                
            term2 = np.abs(x) - np.abs(pop[l]) if fit[i] < fit[l] else np.abs(pop[l]) - np.abs(x)
                
            x_new = np.clip(x + r1 * (best - worst) + r2 * term2, LOWER_BOUND, UPPER_BOUND)
            new_f = obj_func(np.asarray(x_new, dtype=np.float64))
            fes += 1
            if new_f < fit[i]:
                new_pop.append(x_new)
                fit[i] = new_f
            else:
                new_pop.append(x)
        pop = np.array(new_pop)
    return np.min(fit)

def rao3_fe_capped(obj_func):
    pop = initialize_population(POP_SIZE, DIMENSIONS)
    fes = POP_SIZE 
    fit = np.array([obj_func(np.asarray(x, dtype=np.float64)) for x in pop])
    while fes < MAX_FES:
        best, worst = pop[np.argmin(fit)], pop[np.argmax(fit)]
        new_pop = []
        for i, x in enumerate(pop):
            if fes >= MAX_FES: 
                new_pop.append(x)
                continue
            r1, r2 = np.random.rand(DIMENSIONS), np.random.rand(DIMENSIONS)
            l = np.random.randint(0, POP_SIZE)
            while l == i: l = np.random.randint(0, POP_SIZE)
                
            term2 = np.abs(x) - pop[l] if fit[i] < fit[l] else np.abs(pop[l]) - x
                
            x_new = np.clip(x + r1 * (best - np.abs(worst)) + r2 * term2, LOWER_BOUND, UPPER_BOUND)
            new_f = obj_func(np.asarray(x_new, dtype=np.float64))
            fes += 1
            if new_f < fit[i]:
                new_pop.append(x_new)
                fit[i] = new_f
            else:
                new_pop.append(x)
        pop = np.array(new_pop)
    return np.min(fit)

def zo_qrao_fe_capped(obj_func):
    pop = initialize_population(POP_SIZE, DIMENSIONS)
    fes = POP_SIZE
    fit = np.array([obj_func(np.asarray(x, dtype=np.float64)) for x in pop])
    while fes < MAX_FES:
        best, worst = pop[np.argmin(fit)], pop[np.argmax(fit)]
        H = np.argsort(np.argsort(fit)) / (POP_SIZE - 1.0) 
        v_exp = best - worst
        v_dir = v_exp / (np.linalg.norm(v_exp) + 1e-14)
        new_pop = []
        for i, x in enumerate(pop):
            if fes >= MAX_FES: 
                new_pop.append(x)
                continue
            h = H[i] 
            x_class = x + np.random.rand(DIMENSIONS) * v_exp
            u = np.random.randn(DIMENSIONS)
            u_perp = u - np.dot(u, v_dir) * v_dir
            u_perp = u_perp / (np.linalg.norm(u_perp) + 1e-14)
            sigma = np.linalg.norm(x - worst) * np.log(1.0 / (np.random.rand() + 1e-14))
            x_quant = best + (np.random.choice([-1, 1]) * sigma * u_perp)
            x_new = np.clip((h * x_class) + ((1.0 - h) * x_quant), LOWER_BOUND, UPPER_BOUND)
            new_f = obj_func(np.asarray(x_new, dtype=np.float64))
            fes += 1
            if new_f < fit[i]:
                new_pop.append(x_new)
                fit[i] = new_f
            else:
                new_pop.append(x)
        pop = np.array(new_pop)
    return np.min(fit)

# ======================================================
# MULTIPROCESSING WORKER
# ======================================================
def execute_single_run(f_num, bias, target_algo):
    warnings.simplefilter("ignore")
    class_name = f"F{f_num}2014"
    obj_func = getattr(cec2014, class_name)(ndim=DIMENSIONS).evaluate
    
    if target_algo == "RAO2":
        err_base = rao2_fe_capped(obj_func) - bias
    else:
        err_base = rao3_fe_capped(obj_func) - bias
        
    err_zo = zo_qrao_fe_capped(obj_func) - bias
    
    return err_base, err_zo

# ======================================================
# STATISTICAL ENGINE
# ======================================================
def compute_wilcoxon(base_arr, var_arr, base_mean, var_mean):
    if np.array_equal(base_arr, var_arr): 
        return "Tie", 1.000
    try:
        _, p = wilcoxon(base_arr, var_arr)
        if p < 0.05:
            return ("Win", p) if var_mean < base_mean else ("Loss", p)
        return "Tie", p
    except:
        return "Tie", 1.000

# ======================================================
# MAIN EXECUTION SCRIPT
# ======================================================
if __name__ == '__main__':
    # Thermal Safety Valve: Leave 2 cores free for the OS to prevent crashing
    total_cores = multiprocessing.cpu_count()
    safe_cores = max(1, total_cores - 2)
    
    print("=========================================================================================================")
    print(f" INITIATING EXPERIMENT: ZO-QRAO vs {TARGET_ALGORITHM}")
    print(f" Hardware: {safe_cores} CPU Cores Active ({total_cores - safe_cores} reserved for OS stability)")
    print(f" Parameters: {MAX_FES} FEs | {RUNS} Runs | {DIMENSIONS} Dimensions")
    print("=========================================================================================================")
    print(f"{'Func':<4} | {TARGET_ALGORITHM + ' (Mean ± Std)':<28} | {'ZO-QRAO (Mean ± Std)':<28} | {'W/T/L':<6} | {'p-val'}")
    print("-" * 105)

    wins, ties, losses = 0, 0, 0
    all_means_base, all_means_zo = [], []

    for f_num in range(1, 24):
        class_name = f"F{f_num}2014"
        if not hasattr(cec2014, class_name): continue
        
        bias = f_num * 100.0
        err_base_list, err_zo_list = [], []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=safe_cores) as executor:
            futures = [executor.submit(execute_single_run, f_num, bias, TARGET_ALGORITHM) for _ in range(RUNS)]
            for future in concurrent.futures.as_completed(futures):
                e_base, e_zo = future.result()
                err_base_list.append(e_base)
                err_zo_list.append(e_zo)

        m_base, s_base = np.mean(err_base_list), np.std(err_base_list)
        m_zo, s_zo = np.mean(err_zo_list), np.std(err_zo_list)
        
        all_means_base.append(m_base)
        all_means_zo.append(m_zo)
        
        stat, p_val = compute_wilcoxon(err_base_list, err_zo_list, m_base, m_zo)
        if stat == "Win": wins += 1
        elif stat == "Loss": losses += 1
        else: ties += 1

        print(f"F{f_num:<3} | {m_base:.2e} ± {s_base:.2e} {' ' * 8} | {m_zo:.2e} ± {s_zo:.2e} {' ' * 8} | {stat:<6} | {p_val:.4f}")

    # ======================================================
    # FRIEDMAN RANKING CALCULATION
    # ======================================================
    matrix = np.array([all_means_base, all_means_zo])
    ranks = np.array([rankdata(col) for col in matrix.T])
    avg_ranks = np.mean(ranks, axis=0)

    print("=========================================================================================================")
    print(f" FINAL BENCHMARK TALLY: ZO-QRAO vs {TARGET_ALGORITHM}")
    print("=========================================================================================================")
    print(f" Wins - Ties - Losses      : {wins} - {ties} - {losses}")
    print(f" Total Functions Evaluated : {wins + ties + losses}")
    print("-" * 105)
    print(" AVERAGE FRIEDMAN RANKING (Lower is Better)")
    print(f" 1. ZO-QRAO                : {avg_ranks[1]:.2f}")
    print(f" 2. {TARGET_ALGORITHM:<22} : {avg_ranks[0]:.2f}")
    print("=========================================================================================================")