import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.stats import wilcoxon
import opfunu.cec_based.cec2022 as cec2022

# ======================================================
# GENERAL CONFIGURATION & PARAMETERS
# ======================================================
POP_SIZE = 40        
DIMENSIONS = 10      
MAX_ITERATIONS = 300 
RUNS = 15            

LOWER_BOUND = -100.0
UPPER_BOUND = 100.0

def initialize_population(pop_size, dimensions):
    return np.random.uniform(LOWER_BOUND, UPPER_BOUND, (pop_size, dimensions))

def vectorized_levy(pop_size, dimensions, beta=1.5):
    sigma_u = ((math.gamma(1 + beta) * np.sin(np.pi * beta / 2)) / 
               (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
    u = np.random.normal(0, sigma_u, (pop_size, dimensions))
    v = np.random.normal(0, 1, (pop_size, dimensions))
    return u / (np.abs(v) ** (1 / beta))

# ======================================================
# VECTORIZED CLASSICAL RAO-1
# ======================================================
def classical_rao1_cec(obj_func, pop_size, dimensions, max_iter):
    population = initialize_population(pop_size, dimensions)
    convergence = []

    for _ in range(max_iter):
        fitness = np.array([obj_func(ind) for ind in population])
        best = population[np.argmin(fitness)]
        worst = population[np.argmax(fitness)]

        r = np.random.rand(pop_size, 1)
        new_population = population + r * (best - worst)
        new_population = np.clip(new_population, LOWER_BOUND, UPPER_BOUND)
        
        new_fitness = np.array([obj_func(ind) for ind in new_population])
        improved = new_fitness < fitness
        population[improved] = new_population[improved]
        
        convergence.append(np.min(np.minimum(fitness, new_fitness)))

    final_fitness = np.array([obj_func(ind) for ind in population])
    return np.min(final_fitness), convergence

# ======================================================
# TRUE TRAJECTORY LEVY-SCALED QRAO (CORRECTED)
# ======================================================
def levy_geometric_qrao_cec(obj_func, pop_size, dimensions, max_iter):
    population = initialize_population(pop_size, dimensions)
    convergence = []

    for _ in range(max_iter):
        fitness = np.array([obj_func(ind) for ind in population])
        best = population[np.argmin(fitness)]
        worst = population[np.argmax(fitness)]

        # Levy generation paired with uniform random step scaling
        levy_scale = np.random.rand(pop_size, dimensions) * np.abs(np.tanh(vectorized_levy(pop_size, dimensions)))

        # FIX: Base updates on the shrinking (best - worst) trajectory vector
        new_population = population + levy_scale * (best - worst)
        new_population = np.clip(new_population, LOWER_BOUND, UPPER_BOUND)

        new_fitness = np.array([obj_func(ind) for ind in new_population])
        improved = new_fitness < fitness
        population[improved] = new_population[improved]

        convergence.append(np.min(np.minimum(fitness, new_fitness)))

    final_fitness = np.array([obj_func(ind) for ind in population])
    return np.min(final_fitness), convergence

# ======================================================
# RUN EVALUATIONS
# ======================================================
def execute_ieee_cec_comparison(cec_class_instance, function_name, true_bias):
    func_instance = cec_class_instance(ndim=DIMENSIONS)
    obj_func = func_instance.evaluate

    rao_finals, qrao_finals = [], []
    rao_best_curve, qrao_best_curve = None, None

    print(f"\nEvaluating performance on {function_name} over {RUNS} trials...")

    for run in range(RUNS):
        best_fit, curve = classical_rao1_cec(obj_func, POP_SIZE, DIMENSIONS, MAX_ITERATIONS)
        rao_finals.append(best_fit - true_bias)
        if run == 0: rao_best_curve = np.array(curve) - true_bias

    for run in range(RUNS):
        best_fit, curve = levy_geometric_qrao_cec(obj_func, POP_SIZE, DIMENSIONS, MAX_ITERATIONS)
        qrao_finals.append(best_fit - true_bias)
        if run == 0: qrao_best_curve = np.array(curve) - true_bias

    try:
        _, p_value = wilcoxon(rao_finals, qrao_finals)
    except ValueError:
        p_value = 1.0

    print("="*80)
    print(f" TRUE TRAJECTORY LEVY QRAO CEC REPORT: {function_name}")
    print("="*80)
    print(f"Metric Parameters        | Classical Rao-1          | True Trajectory Levy-QRAO")
    print("-"*80)
    print(f"Best Error Reached       | {np.min(rao_finals):<24.4e} | {np.min(qrao_finals):.4e}")
    print(f"Worst Error Reached      | {np.max(rao_finals):<24.4e} | {np.max(qrao_finals):.4e}")
    print(f"Mean Objective Error     | {np.mean(rao_finals):<24.4e} | {np.mean(qrao_finals):.4e}")
    print(f"Standard Deviation (Std) | {np.std(rao_finals):<24.4e} | {np.std(qrao_finals):.4e}")
    print(f"Wilcoxon Significance    | p-value = {p_value:.4e}   | (Alpha Flag = 0.05)")
    print("="*80)

    plt.figure(figsize=(10, 5.5))
    plt.plot(rao_best_curve + 1e-14, label="Classical Rao-1", color="crimson", linewidth=1.8)
    plt.plot(qrao_best_curve + 1e-14, label="True Trajectory Levy QRAO", color="darkviolet", linewidth=2.2)
    plt.yscale("log")
    plt.title(f"True Trajectory Convergence Curve: {function_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Generation Iteration Step Counter")
    plt.ylabel("Log Optimization Error Space [f(x) - Bias]")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

execute_ieee_cec_comparison(cec2022.F12022, "F1: Shifted/Rotated Zakharov", true_bias=300.0)
execute_ieee_cec_comparison(cec2022.F22022, "F2: Shifted/Rotated Rosenbrock", true_bias=400.0)
