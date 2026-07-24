import numpy as np
import time
import warnings
import opfunu.cec_based.cec2014 as cec2014
from scipy.stats import wilcoxon
import concurrent.futures
import multiprocessing

warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURATION
# ======================================================
POP_SIZE = 50        
DIMENSIONS = 30      
MAX_FES = 3000     
RUNS = 10            

LOWER_BOUND = -100.0
UPPER_BOUND = 100.0

def initialize_population(pop_size, dimensions):
    return np.random.uniform(LOWER_BOUND, UPPER_BOUND, (pop_size, dimensions))

# ======================================================
# ALGORITHMS (Identical to previous, compressed for space)
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

def rank_h_only_fe_capped(obj_func):
    pop = initialize_population(POP_SIZE, DIMENSIONS)
    fes = POP_SIZE
    fit = np.array([obj_func(np.asarray(x, dtype=np.float64)) for x in pop])
    sigma_min = (UPPER_BOUND - LOWER_BOUND) / np.sqrt(POP_SIZE)
    while fes < MAX_FES:
        best, worst = pop[np.argmin(fit)], pop[np.argmax(fit)]
        H = np.argsort(np.argsort(fit)) / (POP_SIZE - 1.0) 
        v_exp = best - worst
        new_pop = []
        for i, x in enumerate(pop):
            if fes >= MAX_FES: 
                new_pop.append(x)
                continue
            h = H[i] 
            x_class = x + np.random.rand(DIMENSIONS) * v_exp
            u = np.random.randn(DIMENSIONS)
            u_dir = u / (np.linalg.norm(u) + 1e-14)
            sigma = max(np.linalg.norm(x - worst), sigma_min) * np.log(1.0 / (np.random.rand() + 1e-14))
            x_quant = best + (np.random.choice([-1, 1]) * sigma * u_dir)
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

def rank_h_orthogonal_fe_capped(obj_func):
    pop = initialize_population(POP_SIZE, DIMENSIONS)
    fes = POP_SIZE
    fit = np.array([obj_func(np.asarray(x, dtype=np.float64)) for x in pop])
    sigma_min = (UPPER_BOUND - LOWER_BOUND) / np.sqrt(POP_SIZE)
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
            sigma = max(np.linalg.norm(x - worst), sigma_min) * np.log(1.0 / (np.random.rand() + 1e-14))
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
# PARALLEL EXECUTION WORKER
# ======================================================
def execute_single_run(f_num, bias):
    # Initialize function inside the worker to prevent memory lock errors
    class_name = f"F{f_num}2014"
    obj_func = getattr(cec2014, class_name)(ndim=DIMENSIONS).evaluate
    
    r_err = rao1_fe_capped(obj_func) - bias
    rk_err = rank_h_only_fe_capped(obj_func) - bias
    ort_err = rank_h_orthogonal_fe_capped(obj_func) - bias
    
    return r_err, rk_err, ort_err

def eval_wtl(base_mean, var_mean, base_arr, var_arr):
    if np.array_equal(base_arr, var_arr): return "Tie"
    try:
        _, p = wilcoxon(base_arr, var_arr)
        if p < 0.05:
            return "Win" if var_mean < base_mean else "Loss"
        return "Tie"
    except:
        return "Tie"

# ======================================================
# MAIN MULTIPROCESSING ENGINE
# ======================================================
if __name__ == '__main__':
    # Automatically detect CPU cores
    available_cores = multiprocessing.cpu_count()
    print(f"INITIATING FULL CEC 2014 SUITE: F1 - F23")
    print(f"Hardware: Utilizing {available_cores} CPU Cores via Multiprocessing")
    print(f"Parameters: {MAX_FES} FEs | {RUNS} Runs | {DIMENSIONS}D")
    print("=" * 125)
    print(f"{'Func':<5} | {'Classical Rao (Mean ± Std)':<30} | {'Rank-H Only (Mean ± Std)':<30} | {'Rank+Ortho (Mean ± Std)':<30} | {'W/T/L'}")
    print("-" * 125)

    wins, ties, losses = 0, 0, 0
    
    # Process each function sequentially, but run its 10 trials in parallel
    for f_num in range(1, 24):
        class_name = f"F{f_num}2014"
        if not hasattr(cec2014, class_name): continue
        
        bias = f_num * 100.0
        err_rao, err_rank, err_ortho = [], [], []
        
        # Deploy parallel workers for the 10 runs
        with concurrent.futures.ProcessPoolExecutor(max_workers=available_cores) as executor:
            futures = [executor.submit(execute_single_run, f_num, bias) for _ in range(RUNS)]
            
            for future in concurrent.futures.as_completed(futures):
                r_e, rk_e, ort_e = future.result()
                err_rao.append(r_e)
                err_rank.append(rk_e)
                err_ortho.append(ort_e)

        m_rao, s_rao = np.mean(err_rao), np.std(err_rao)
        m_rnk, s_rnk = np.mean(err_rank), np.std(err_rank)
        m_ort, s_ort = np.mean(err_ortho), np.std(err_ortho)
        
        wtl = eval_wtl(m_rao, m_ort, err_rao, err_ortho)
        if wtl == "Win": wins += 1
        elif wtl == "Loss": losses += 1
        else: ties += 1

        print(f"F{f_num:<3} | {m_rao:.2e} ± {s_rao:.2e} {' ' * 10} | {m_rnk:.2e} ± {s_rnk:.2e} {' ' * 10} | {m_ort:.2e} ± {s_ort:.2e} {' ' * 10} | {wtl}")

    print("=" * 125)
    print("FINAL BENCHMARK TALLY (Rank-H + Ortho vs Classical Rao)")
    print(f"TOTAL WINS:   {wins}")
    print(f"TOTAL TIES:   {ties}")
    print(f"TOTAL LOSSES: {losses}")
    print("=" * 125)