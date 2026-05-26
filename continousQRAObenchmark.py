import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.stats import wilcoxon
import opfunu.cec_based.cec2022 as cec2022

# ======================================================
# IEEE CEC BENCHMARK CONFIGURATION
# ======================================================
POP_SIZE = 50        # Set to match your default function parameter
DIMENSIONS = 10      # Supported CEC 2022 Dimensions: 10 or 20
MAX_ITERATIONS = 300 # Set to match your default function parameter
RUNS = 15            # Number of independent statistical trials

LOWER_BOUND = -100.0
UPPER_BOUND = 100.0

def initialize_population(pop_size, dimensions):
    return np.random.uniform(LOWER_BOUND, UPPER_BOUND, (pop_size, dimensions))

# ======================================================
# NATIVE LEVY COEFFICIENT GENERATOR
# ======================================================
def levy_coefficient(beta=1.5):
    """Generates a single scalar Levy step using Mantegna's algorithm."""
    sigma_u = ((math.gamma(1 + beta) * np.sin(np.pi * beta / 2)) / 
               (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
    u = np.random.normal(0, sigma_u)
    v = np.random.normal(0, 1)
    step = u / (abs(v) ** (1 / beta))
    return step

# ======================================================
# CLASSICAL RAO-1 BASELINE
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
# YOUR EXACT LEVY_QRAO ALGORITHM (UNTOUCHED)
# ======================================================
def levy_qrao(objective_function, pop_size=50, dimensions=30, iterations=300):
    population = initialize_population(pop_size, dimensions)
    convergence = []

    for iteration in range(iterations):
        fitness = np.array([objective_function(x) for x in population])
        best = population[np.argmin(fitness)]
        worst = population[np.argmax(fitness)]
        new_population = []

        for x in population:
            # Levy-distributed coefficient
            r = levy_coefficient()
            # Optional clipping for stability
            r = np.clip(r, -2, 2)
            # Levy-QRAO update
            x_new = x + r * (best - worst)
            x_new = np.clip(x_new, LOWER_BOUND, UPPER_BOUND) # Safety bound constraint

            # Greedy Selection
            if objective_function(x_new) < objective_function(x):
                new_population.append(x_new)
            else:
                new_population.append(x)

        population = np.array(new_population)
        updated_fitness = np.array([objective_function(x) for x in population])
        convergence.append(np.min(updated_fitness))

    return np.min(updated_fitness), convergence

# ======================================================
# EXECUTION & STATISTICAL METRICS FRAMEWORK
# ======================================================
def run_ieee_cec_benchmark(cec_class, function_name, true_bias):
    func_instance = cec_class(ndim=DIMENSIONS)
    obj_func = func_instance.evaluate

    rao_finals, qrao_finals = [], []
    rao_best_curve, qrao_best_curve = None, None

    print(f"\nRunning Head-to-Head Comparison on {function_name} ({DIMENSIONS}D)...")

    # Profile Classical Rao
    for run in range(RUNS):
        best_fit, curve = rao1(obj_func, POP_SIZE, DIMENSIONS, MAX_ITERATIONS)
        rao_finals.append(best_fit - true_bias)
        if run == 0: rao_best_curve = np.array(curve) - true_bias

    # Profile Your Levy QRAO
    for run in range(RUNS):
        best_fit, curve = levy_qrao(obj_func, POP_SIZE, DIMENSIONS, MAX_ITERATIONS)
        qrao_finals.append(best_fit - true_bias)
        if run == 0: qrao_best_curve = np.array(curve) - true_bias

    # Statistical Significance (Wilcoxon Rank-Sum Test)
    try:
        _, p_value = wilcoxon(rao_finals, qrao_finals)
    except ValueError:
        p_value = 1.0

    # Output Formal Results Table
    print("="*80)
    print(f" IEEE CEC COMPETITION REPORT: {function_name}")
    print("="*80)
    print(f"Metric Parameters        | Classical Rao-1          | Your Levy_QRAO")
    print("-"*80)
    print(f"Best Error Reached       | {np.min(rao_finals):<24.4e} | {np.min(qrao_finals):.4e}")
    print(f"Worst Error Reached      | {np.max(rao_finals):<24.4e} | {np.max(qrao_finals):.4e}")
    print(f"Mean Objective Error     | {np.mean(rao_finals):<24.4e} | {np.mean(qrao_finals):.4e}")
    print(f"Standard Deviation (Std) | {np.std(rao_finals):<24.4e} | {np.std(qrao_finals):.4e}")
    print(f"Wilcoxon Significance    | p-value = {p_value:.4e}   | (Alpha Flag = 0.05)")
    print("="*80)

    # Convergence Plot Setup
    plt.figure(figsize=(10, 5.5))
    plt.plot(rao_best_curve + 1e-14, label="Classical Rao-1", color="crimson", linewidth=1.8)
    plt.plot(qrao_best_curve + 1e-14, label="Your Levy_QRAO", color="darkviolet", linewidth=2.2)
    plt.yscale("log")
    plt.title(f"IEEE CEC Convergence Comparison: {function_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Generation Iteration Step")
    plt.ylabel("Log Optimization Error Space [f(x) - Bias]")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run the benchmark suite on Unimodal and Multimodal functions
run_ieee_benchmark_suite = True
if run_ieee_benchmark_suite:
    execute_ieee_cec_comparison = run_ieee_cec_benchmark
    execute_ieee_cec_comparison(cec2022.F12022, "F1: Shifted/Rotated Zakharov", true_bias=300.0)
    execute_ieee_cec_comparison(cec2022.F22022, "F2: Shifted/Rotated Rosenbrock", true_bias=400.0)
