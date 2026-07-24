import os
import warnings

# ======================================================
# ABSOLUTE SILENCE PROTOCOL
# Forces Windows and all sub-processes to mute warnings
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
# IEEE STANDARD CONFIGURATION
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
# ALGORITHM 1: Classical Rao (Baseline)
# ======================================================
def rao1_fe_capped(obj_func):
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
            r = np.random.rand(DIMENSIONS)
            x_new = np.clip(x + r * (best - worst), LOWER_BOUND, UPPER_BOUND)
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
# ALGORITHM 2: ZO-QRAO (Rank-H + Ortho + No Floor)
# ======================================================
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
def execute_single_run(f_num, bias):
    # Double-lock to ensure the worker process is totally silent
    warnings.simplefilter("ignore") 
    class_name = f"F{f_num}2014"
    obj_func = getattr(cec2014, class_name)(ndim=DIMENSIONS).evaluate
    
    err_rao = rao1_fe_capped(obj_func) - bias
    err_zo = zo_qrao_fe_capped(obj_func) - bias
    
    return err_rao, err_zo

# ======================================================
# STATISTICAL ENGINE
# ======================================================
def compute_wilcoxon(base_arr, var_arr, base_mean, var_mean):
    if np.array_equal(base_arr, var_arr): 
        return "Tie", 1.000
    try:
        _, p = wilcoxon(base_arr, var_arr)
        if p < 0.05:
            if var_mean < base_mean:
                return "Win", p
            else:
                return "Loss", p
        return "Tie", p
    except:
        return "Tie", 1.000

# ======================================================
# MAIN EXECUTION SCRIPT
# ======================================================
if __name__ == '__main__':
    available_cores = multiprocessing.cpu_count()
    
    print("=====================================================================================================")
    print(" INITIATING FINAL VALIDATION: ZO-QRAO vs CLASSICAL RAO")
    print(f" Hardware: {available_cores} CPU Cores | Suite: CEC 2014 (F1-F23)")
    print(f" Parameters: {MAX_FES} FEs | {RUNS} Runs | {DIMENSIONS} Dimensions")
    print("=====================================================================================================")
    print(f"{'Func':<5} | {'Classical Rao (Mean ± Std)':<30} | {'ZO-QRAO (Mean ± Std)':<30} | {'W/T/L':<5} | {'p-value'}")
    print("-" * 101)

    wins, ties, losses = 0, 0, 0
    all_means_rao = []
    all_means_zo = []

    for f_num in range(1, 24):
        class_name = f"F{f_num}2014"
        if not hasattr(cec2014, class_name): continue
        
        bias = f_num * 100.0
        err_rao_list = []
        err_zo_list = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=available_cores) as executor:
            futures = [executor.submit(execute_single_run, f_num, bias) for _ in range(RUNS)]
            for future in concurrent.futures.as_completed(futures):
                e_rao, e_zo = future.result()
                err_rao_list.append(e_rao)
                err_zo_list.append(e_zo)

        m_rao, s_rao = np.mean(err_rao_list), np.std(err_rao_list)
        m_zo, s_zo = np.mean(err_zo_list), np.std(err_zo_list)
        
        # Collect for overall Friedman Ranking
        all_means_rao.append(m_rao)
        all_means_zo.append(m_zo)
        
        status, p_val = compute_wilcoxon(err_rao_list, err_zo_list, m_rao, m_zo)
        
        if status == "Win": wins += 1
        elif status == "Loss": losses += 1
        else: ties += 1

        print(f"F{f_num:<3} | {m_rao:.2e} ± {s_rao:.2e} {' ' * 10} | {m_zo:.2e} ± {s_zo:.2e} {' ' * 10} | {status:<5} | {p_val:.4f}")

    # ======================================================
    # FRIEDMAN RANKING CALCULATION
    # ======================================================
    matrix = np.array([all_means_rao, all_means_zo])
    ranks = np.array([rankdata(col) for col in matrix.T])
    avg_ranks = np.mean(ranks, axis=0)

    print("=====================================================================================================")
    print(" FINAL BENCHMARK TALLY & STATISTICAL SCORECARD")
    print("=====================================================================================================")
    print(f" ZO-QRAO vs Rao (Wins - Ties - Losses) : {wins} - {ties} - {losses}")
    print(f" Total Functions Evaluated             : {wins + ties + losses}")
    print("-" * 101)
    print(" AVERAGE FRIEDMAN RANKING (Lower is Better)")
    print(f" 1. ZO-QRAO       : {avg_ranks[1]:.2f}")
    print(f" 2. Classical Rao : {avg_ranks[0]:.2f}")
    print("=====================================================================================================")