import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.stats import wilcoxon
import opfunu.cec_based.cec2022 as cec2022

# ======================================================
# GLOBAL BENCHMARK CONFIGURATION
# ======================================================
POP_SIZE = 50        
DIMENSIONS = 10      
MAX_ITERATIONS = 300 
RUNS = 15            

LOWER_BOUND = -100.0
UPPER_BOUND = 100.0

def initialize_population(pop_size, dimensions):
    return np.random.uniform(LOWER_BOUND, UPPER_BOUND, (pop_size, dimensions))

def levy_coefficient(beta=1.5):
    sigma_u = ((math.gamma(1 + beta) * np.sin(np.pi * beta / 2)) / 
               (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
    u = np.random.normal(0, sigma_u)
    v = np.random.normal(0, 1)
    return u / (abs(v) ** (1 / beta))

# ======================================================
# ALGORITHM 1: CLASSICAL RAO-1
# ======================================================
def rao1(objective_function, pop_size=50, dimensions=30, iterations=300):
    population = initialize_population(pop_size, dimensions)
    convergence = []

    for iteration in range(iterations):
        fitness = np.array([objective_function(x) for x in population])
        best = population[np.argmin(fitness)]
        worst = population[np.argmax(fitness)]
        new_population = []

        for x in population:
            r = np.random.rand()
            x_new = x + r * (best - worst)
            x_new = np.clip(x_new, LOWER_BOUND, UPPER_BOUND)

            if objective_function(x_new) < objective_function(x):
                new_population.append(x_new)
            else:
                new_population.append(x)

        population = np.array(new_population)
        updated_fitness = np.array([objective_function(x) for x in population])
        convergence.append(np.min(updated_fitness))

    return np.min(updated_fitness), convergence

# ======================================================
# ALGORITHM 2: YOUR ORIGINAL SCALAR LEVY_QRAO
# ======================================================
def original_levy_qrao(objective_function, pop_size=50, dimensions=30, iterations=300):
    population = initialize_population(pop_size, dimensions)
    convergence = []

    for iteration in range(iterations):
        fitness = np.array([objective_function(x) for x in population])
        best = population[np.argmin(fitness)]
        worst = population[np.argmax(fitness)]
        new_population = []

        for x in population:
            r = levy_coefficient()
            r = np.clip(r, -2, 2)
            
            x_new = x + r * (best - worst)
            x_new = np.clip(x_new, LOWER_BOUND, UPPER_BOUND)

            if objective_function(x_new) < objective_function(x):
                new_population.append(x_new)
            else:
                new_population.append(x)

        population = np.array(new_population)
        updated_fitness = np.array([objective_function(x) for x in population])
        convergence.append(np.min(updated_fitness))

    return np.min(updated_fitness), convergence

# ======================================================
# ALGORITHM 3: UPGRADED ADAPTIVE SCALAR LEVY_QRAO
# ======================================================
def adaptive_scalar_levy_qrao(objective_function, pop_size=50, dimensions=30, iterations=300):
    population = initialize_population(pop_size, dimensions)
    convergence = []

    for iteration in range(iterations):
        fitness = np.array([objective_function(x) for x in population])
        best = population[np.argmin(fitness)]
        worst = population[np.argmax(fitness)]
        new_population = []

        # Step scaling factor that naturally narrows as generation depth increases
        alpha = 1.0 - (iteration / iterations)

        for x in population:
            # Preserves coordinated vector step while dampening step size over time
            r = levy_coefficient() * alpha
            r = np.clip(r, -2, 2)
            
            x_new = x + r * (best - worst)
            x_new = np.clip(x_new, LOWER_BOUND, UPPER_BOUND)

            if objective_function(x_new) < objective_function(x):
                new_population.append(x_new)
            else:
                new_population.append(x)

        population = np.array(new_population)
        updated_fitness = np.array([objective_function(x) for x in population])
        convergence.append(np.min(updated_fitness))

    return np.min(updated_fitness), convergence

# ======================================================
# PERFORMANCE MONITOR ENGINE
# ======================================================
def run_ieee_cec_triple_comparison(cec_class, function_name, true_bias):
    func_instance = cec_class(ndim=DIMENSIONS)
    obj_func = func_instance.evaluate

    rao_finals, original_qrao_finals, adaptive_qrao_finals = [], [], []
    rao_curve, original_qrao_curve, adaptive_qrao_curve = None, None, None

    print(f"\nEvaluating performance on {function_name} ({DIMENSIONS}D)...")

    for run in range(RUNS):
        best_fit, curve = rao1(obj_func, POP_SIZE, DIMENSIONS, MAX_ITERATIONS)
        rao_finals.append(best_fit - true_bias)
        if run == 0: rao_curve = np.array(curve) - true_bias

    for run in range(RUNS):
        best_fit, curve = original_levy_qrao(obj_func, POP_SIZE, DIMENSIONS, MAX_ITERATIONS)
        original_qrao_finals.append(best_fit - true_bias)
        if run == 0: original_qrao_curve = np.array(curve) - true_bias

    for run in range(RUNS):
        best_fit, curve = adaptive_scalar_levy_qrao(obj_func, POP_SIZE, DIMENSIONS, MAX_ITERATIONS)
        adaptive_qrao_finals.append(best_fit - true_bias)
        if run == 0: adaptive_qrao_curve = np.array(curve) - true_bias

    try:
        _, p_value_adaptive_vs_orig = wilcoxon(adaptive_qrao_finals, original_qrao_finals)
    except ValueError:
        p_value_adaptive_vs_orig = 1.0

    print("="*95)
    print(f" OFFICIAL IEEE SCALAR ADVANCEMENT REPORT: {function_name}")
    print("="*95)
    print(f"Metric Parameters        | Classical Rao-1          | Original Scalar QRAO     | Adaptive Scalar QRAO")
    print("-"*95)
    print(f"Best Error Reached       | {np.min(rao_finals):<24.4e} | {np.min(original_qrao_finals):<24.4e} | {np.min(adaptive_qrao_finals):.4e}")
    print(f"Worst Error Reached      | {np.max(rao_finals):<24.4e} | {np.max(original_qrao_finals):<24.4e} | {np.max(adaptive_qrao_finals):.4e}")
    print(f"Mean Objective Error     | {np.mean(rao_finals):<24.4e} | {np.mean(original_qrao_finals):<24.4e} | {np.mean(adaptive_qrao_finals):.4e}")
    print(f"Standard Deviation (Std) | {np.std(rao_finals):<24.4e} | {np.std(original_qrao_finals):<24.4e} | {np.std(adaptive_qrao_finals):.4e}")
    print(f"Adaptive vs Original Sig. | ------------------------ | ------------------------ | p-value = {p_value_adaptive_vs_orig:.4e}")
    print("="*95)

    plt.figure(figsize=(11, 6))
    plt.plot(rao_curve + 1e-14, label="Classical Rao-1", color="crimson", linewidth=1.8)
    plt.plot(original_qrao_curve + 1e-14, label="Original Scalar Levy_QRAO", color="orange", linestyle="--", linewidth=1.8)
    plt.plot(adaptive_qrao_curve + 1e-14, label="Adaptive Scalar Levy_QRAO (Preserved Trajectory)", color="dodgerblue", linewidth=2.5)
    
    plt.yscale("log")
    plt.title(f"IEEE CEC Step-Scale Adaptation Evaluation: {function_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Generation Iteration Step")
    plt.ylabel("Log Optimization Error Space [f(x) - Bias]")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run the comparative benchmarks
run_ieee_cec_triple_comparison(cec2022.F12022, "F1: Shifted/Rotated Zakharov", true_bias=300.0)
run_ieee_cec_triple_comparison(cec2022.F22022, "F2: Shifted/Rotated Rosenbrock", true_bias=400.0)
