import os
import warnings
import csv
from datetime import datetime
import numpy as np
import opfunu.cec_based.cec2022 as cec2022
from scipy.stats import wilcoxon, rankdata
import concurrent.futures
import multiprocessing

# ======================================================
# ABSOLUTE SILENCE PROTOCOL
# ======================================================
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ======================================================
# EXPERIMENT CONFIGURATION (IEEE CEC 2022 Standard)
# ======================================================
POP_SIZE = 50        
DIMENSIONS = 20      
MAX_FES = 1000000    # Official Deep Search Budget for 20D
RUNS = 30            
LOWER_BOUND = -100.0
UPPER_BOUND = 100.0

# CEC 2022 Official Optimal Biases (F1 - F12)
CEC2022_BIAS = {
    1: 300.0, 2: 400.0, 3: 600.0, 4: 800.0, 5: 900.0, 6: 1400.0,
    7: 2000.0, 8: 2200.0, 9: 2300.0, 10: 2400.0, 11: 2600.0, 12: 2700.0
}

def initialize_population(pop_size, dimensions):
    return np.random.uniform(LOWER_BOUND, UPPER_BOUND, (pop_size, dimensions))

# ======================================================
# ALGORITHMS (With IEEE 1e-8 Early Termination & Seeds)
# ======================================================
def rao1_fe_capped(obj_func, bias, run_seed):
    np.random.seed(run_seed) # Reproducible randomness
    pop = initialize_population(POP_SIZE, DIMENSIONS)
    fes = POP_SIZE 
    fit = np.array([obj_func(np.asarray(x, dtype=np.float64)) for x in pop])
    
    while fes < MAX_FES:
        best_idx, worst_idx = np.argmin(fit), np.argmax(fit)
        
        # IEEE Early Termination Protocol
        if (fit[best_idx] - bias) <= 1e-8:
            return 0.0

        best, worst = pop[best_idx], pop[worst_idx]
        new_pop = []
        for i, x in enumerate(pop):
            if fes >= MAX_FES: break
                
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
        
    final_error = np.min(fit) - bias
    return max(0.0, final_error) if final_error <= 1e-8 else final_error

def zo_qrao_fe_capped(obj_func, bias, run_seed):
    np.random.seed(run_seed) # Reproducible randomness
    pop = initialize_population(POP_SIZE, DIMENSIONS)
    fes = POP_SIZE
    fit = np.array([obj_func(np.asarray(x, dtype=np.float64)) for x in pop])
    
    while fes < MAX_FES:
        best_idx, worst_idx = np.argmin(fit), np.argmax(fit)
        
        # IEEE Early Termination Protocol
        if (fit[best_idx] - bias) <= 1e-8:
            return 0.0

        best, worst = pop[best_idx], pop[worst_idx]
        H = np.argsort(np.argsort(fit)) / (POP_SIZE - 1.0) 
        v_exp = best - worst
        v_dir = v_exp / (np.linalg.norm(v_exp) + 1e-14)
        new_pop = []
        for i, x in enumerate(pop):
            if fes >= MAX_FES: break
                
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
        
    final_error = np.min(fit) - bias
    return max(0.0, final_error) if final_error <= 1e-8 else final_error

# ======================================================
# MULTIPROCESSING WORKER
# ======================================================
def execute_single_run(f_num, run_id):
    warnings.simplefilter("ignore")
    class_name = f"F{f_num}2022"
    obj_func = getattr(cec2022, class_name)(ndim=DIMENSIONS).evaluate
    bias = CEC2022_BIAS[f_num]
    
    # Generate a deterministic seed so both algorithms play on a level field
    seed_base = 202200 + run_id
    
    err_base = rao1_fe_capped(obj_func, bias, seed_base)
    err_zo = zo_qrao_fe_capped(obj_func, bias, seed_base)
    
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
    total_cores = multiprocessing.cpu_count()
    safe_cores = max(1, total_cores - 2)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"IEEE_CEC2022_RAO1_Results_{timestamp}.csv"
    
    # 1. LIVE SAVING: Write CSV Headers immediately
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Function", 
            "Rao-1_Best", "Rao-1_Worst", "Rao-1_Median", "Rao-1_Mean", "Rao-1_Std",
            "ZO-QRAO_Best", "ZO-QRAO_Worst", "ZO-QRAO_Median", "ZO-QRAO_Mean", "ZO-QRAO_Std",
            "W/T/L", "p-value"
        ])

    print("=========================================================================================================")
    print(" INITIATING IEEE CEC 2022 BENCHMARK: ZO-QRAO vs CLASSICAL RAO-1")
    print(f" Hardware: {safe_cores} CPU Cores Active | LIVE SAVING to: {filename}")
    print(f" Parameters: {MAX_FES} FEs | {RUNS} Runs | {DIMENSIONS} Dimensions | IEEE 1e-8 Cutoff Active")
    print("=========================================================================================================")
    print(f"{'Func':<4} | {'Rao-1 (Mean ± Std)':<28} | {'ZO-QRAO (Mean ± Std)':<28} | {'W/T/L':<6} | {'p-val'}")
    print("-" * 105)

    wins, ties, losses = 0, 0, 0
    all_means_base, all_means_zo = [], []

    with concurrent.futures.ProcessPoolExecutor(max_workers=safe_cores) as executor:
        for f_num in range(1, 13):
            class_name = f"F{f_num}2022"
            if not hasattr(cec2022, class_name): continue
            
            err_base_list, err_zo_list = [], []
            
            # Pass run_id to ensure seed consistency
            futures = [executor.submit(execute_single_run, f_num, r) for r in range(RUNS)]
            for future in concurrent.futures.as_completed(futures):
                e_base, e_zo = future.result()
                err_base_list.append(e_base)
                err_zo_list.append(e_zo)

            # IEEE Required Statistics
            b_best, b_worst, b_median, b_mean, b_std = np.min(err_base_list), np.max(err_base_list), np.median(err_base_list), np.mean(err_base_list), np.std(err_base_list)
            z_best, z_worst, z_median, z_mean, z_std = np.min(err_zo_list), np.max(err_zo_list), np.median(err_zo_list), np.mean(err_zo_list), np.std(err_zo_list)
            
            all_means_base.append(b_mean)
            all_means_zo.append(z_mean)
            
            stat, p_val = compute_wilcoxon(err_base_list, err_zo_list, b_mean, z_mean)
            if stat == "Win": wins += 1
            elif stat == "Loss": losses += 1
            else: ties += 1

            # Print concise summary to terminal
            print(f"F{f_num:<3} | {b_mean:.2e} ± {b_std:.2e} {' ' * 8} | {z_mean:.2e} ± {z_std:.2e} {' ' * 8} | {stat:<6} | {p_val:.4f}")
            
            # 2. LIVE SAVING: Append detailed stats to CSV after each function finishes
            with open(filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    f"F{f_num}", 
                    f"{b_best:.4e}", f"{b_worst:.4e}", f"{b_median:.4e}", f"{b_mean:.4e}", f"{b_std:.4e}",
                    f"{z_best:.4e}", f"{z_worst:.4e}", f"{z_median:.4e}", f"{z_mean:.4e}", f"{z_std:.4e}",
                    stat, f"{p_val:.4f}"
                ])

    # ======================================================
    # FRIEDMAN RANKING CALCULATION
    # ======================================================
    matrix = np.array([all_means_base, all_means_zo])
    ranks = np.array([rankdata(col) for col in matrix.T])
    avg_ranks = np.mean(ranks, axis=0)
    
    # Append summary data to the bottom of the Excel file
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([]) 
        writer.writerow(["SUMMARY STATISTICS"])
        writer.writerow(["ZO-QRAO vs Rao-1 (Wins/Ties/Losses)", f"{wins} / {ties} / {losses}"])
        writer.writerow(["Total Functions Evaluated", f"{wins + ties + losses}"])
        writer.writerow([])
        writer.writerow(["AVERAGE FRIEDMAN RANKING (Lower is Better)"])
        writer.writerow(["1. ZO-QRAO", f"{avg_ranks[1]:.2f}"])
        writer.writerow(["2. Rao-1", f"{avg_ranks[0]:.2f}"])

    print("=========================================================================================================")
    print(" FINAL CEC 2022 TALLY: ZO-QRAO vs CLASSICAL RAO-1")
    print("=========================================================================================================")
    print(f" Wins - Ties - Losses      : {wins} - {ties} - {losses}")
    print(f" Total Functions Evaluated : {wins + ties + losses}")
    print("-" * 105)
    print(" AVERAGE FRIEDMAN RANKING (Lower is Better)")
    print(f" 1. ZO-QRAO                : {avg_ranks[1]:.2f}")
    print(f" 2. Rao-1                  : {avg_ranks[0]:.2f}")
    print("=========================================================================================================")