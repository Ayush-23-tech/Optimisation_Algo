import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import opfunu.cec_based.cec2022 as cec2022
import concurrent.futures
import multiprocessing

# ======================================================
# ABSOLUTE SILENCE PROTOCOL
# ======================================================
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# ======================================================
# GRAPHING CONFIGURATION 
# ======================================================
TARGET_FUNCTION = 8
POP_SIZE = 50        
DIMENSIONS = 20      
MAX_FES = 1000000        # Official CEC 2022 Deep Search Budget
RUNS = 30                
LOWER_BOUND = -100.0
UPPER_BOUND = 100.0
RECORD_INTERVAL = MAX_FES // 100  

CEC2022_BIAS = {
    1: 300.0, 2: 400.0, 3: 600.0, 4: 800.0, 5: 900.0, 6: 1400.0,
    7: 2000.0, 8: 2200.0, 9: 2300.0, 10: 2400.0, 11: 2600.0, 12: 2700.0
}

def initialize_population(pop_size, dimensions):
    return np.random.uniform(LOWER_BOUND, UPPER_BOUND, (pop_size, dimensions))

# ======================================================
# ALGORITHMS (MODIFIED FOR HISTORY TRACKING & EARLY STOP)
# ======================================================
def rao1_convergence(obj_func, bias):
    pop = initialize_population(POP_SIZE, DIMENSIONS)
    fes = POP_SIZE 
    fit = np.array([obj_func(np.asarray(x, dtype=np.float64)) for x in pop])
    
    best_so_far = np.min(fit)
    history = [best_so_far - bias]
    
    while fes < MAX_FES:
        if (best_so_far - bias) <= 1e-8:
            break # IEEE Early Termination Rule
            
        best, worst = pop[np.argmin(fit)], pop[np.argmax(fit)]
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
                
            current_min = np.min(fit)
            if current_min < best_so_far: best_so_far = current_min
            if fes % RECORD_INTERVAL == 0:
                history.append(best_so_far - bias)
                
        pop = np.array(new_pop)
        
    # Pad the rest of the history array if it terminated early
    while len(history) <= 100:
        history.append(best_so_far - bias)
        
    return history

def zo_qrao_convergence(obj_func, bias):
    pop = initialize_population(POP_SIZE, DIMENSIONS)
    fes = POP_SIZE
    fit = np.array([obj_func(np.asarray(x, dtype=np.float64)) for x in pop])
    
    best_so_far = np.min(fit)
    history = [best_so_far - bias]
    
    while fes < MAX_FES:
        if (best_so_far - bias) <= 1e-8:
            break # IEEE Early Termination Rule
            
        best, worst = pop[np.argmin(fit)], pop[np.argmax(fit)]
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
                
            current_min = np.min(fit)
            if current_min < best_so_far: best_so_far = current_min
            if fes % RECORD_INTERVAL == 0:
                history.append(best_so_far - bias)
                
        pop = np.array(new_pop)
        
    # Pad the rest of the history array if it terminated early
    while len(history) <= 100:
        history.append(best_so_far - bias)
        
    return history

# ======================================================
# MULTIPROCESSING WORKER
# ======================================================
def execute_single_run(f_num):
    warnings.simplefilter("ignore")
    class_name = f"F{f_num}2022"
    obj_func = getattr(cec2022, class_name)(ndim=DIMENSIONS).evaluate
    bias = CEC2022_BIAS[f_num]
    
    hist_base = rao1_convergence(obj_func, bias)
    hist_zo = zo_qrao_convergence(obj_func, bias)
    
    return hist_base, hist_zo

# ======================================================
# MAIN EXECUTION & IEEE PLOTTING ENGINE
# ======================================================
if __name__ == '__main__':
    total_cores = multiprocessing.cpu_count()
    safe_cores = max(1, total_cores - 2)
    
    print("=======================================================================")
    print(f" GENERATING IEEE-FORMATTED GRAPH FOR F{TARGET_FUNCTION} (ZO-QRAO vs Rao-1)")
    print(f" Simulating {RUNS} Runs across {safe_cores} Cores...")
    print("=======================================================================")

    all_hist_base = []
    all_hist_zo = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=safe_cores) as executor:
        futures = [executor.submit(execute_single_run, TARGET_FUNCTION) for _ in range(RUNS)]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            h_base, h_zo = future.result()
            
            if len(h_base) > 101: h_base = h_base[:101]
            if len(h_zo) > 101: h_zo = h_zo[:101]
                
            all_hist_base.append(h_base)
            all_hist_zo.append(h_zo)
            print(f" Run {i+1}/{RUNS} Completed.")

    mean_hist_base = np.mean(all_hist_base, axis=0)
    mean_hist_zo = np.mean(all_hist_zo, axis=0)
    x_axis = np.linspace(0, MAX_FES, len(mean_hist_base))

    # ======================================================
    # IEEE JOURNAL QUALITY MATPLOTLIB SETTINGS
    # ======================================================
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 13
    mpl.rcParams['legend.fontsize'] = 11

    # Standard IEEE column width is usually ~3.5 inches. 
    # (7, 5) gives a highly readable aspect ratio when scaled down in Word/LaTeX.
    plt.figure(figsize=(7, 5), dpi=600) 
    
    # We clip values below 1e-8 to cleanly represent the IEEE stopping criteria
    mean_hist_base = np.clip(mean_hist_base, 1e-8, None)
    mean_hist_zo = np.clip(mean_hist_zo, 1e-8, None)

    # Plot the lines with clear visual distinctions
    plt.semilogy(x_axis, mean_hist_base, label='Classical Rao-1', color='#1f77b4', linestyle='--', linewidth=2.2)
    plt.semilogy(x_axis, mean_hist_zo, label='ZO-QRAO (Proposed)', color='#d62728', linestyle='-', linewidth=2.2)

    # Clean, professional titles and labels
    plt.title(f"Convergence Graph: CEC 2022 F{TARGET_FUNCTION} (20D)")
    plt.xlabel('Number of Function Evaluations (FEs)')
    plt.ylabel('Log Error Value: f(x) - f(x*)')
    
    # Inner Margins (Prevents lines from touching the box)
    plt.margins(x=0.02, y=0.05) 
    
    # Formatted Grid (Softer lines, less distracting)
    plt.grid(True, which="major", color='gray', linestyle='-', alpha=0.3)
    plt.grid(True, which="minor", color='gray', linestyle=':', alpha=0.15)
    
    # Legend formatting
    plt.legend(loc='upper right', framealpha=1.0, edgecolor='black', fancybox=False)
    
    # Outer Margins (Adds padding so labels are never cut off)
    plt.tight_layout(pad=2.0)

    # Save the file
    filename = f"IEEE_Convergence_F{TARGET_FUNCTION}_Rao1.png"
    plt.savefig(filename, bbox_inches='tight')
    
    print("=======================================================================")
    print(f" SUCCESS! High-Res Graph saved to your folder as: {filename}")
    print("=======================================================================")
    
    plt.show()