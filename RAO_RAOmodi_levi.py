import numpy as np
import matplotlib.pyplot as plt
import math


# ======================================================
# BENCHMARK FUNCTIONS
# ======================================================

def sphere(x):

    return np.sum(x**2)


def rastrigin(x):

    n = len(x)

    return (
        10*n
        + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    )


def ackley(x):

    n = len(x)

    term1 = -20 * np.exp(
        -0.2 * np.sqrt(np.sum(x**2)/n)
    )

    term2 = -np.exp(
        np.sum(np.cos(2*np.pi*x))/n
    )

    return term1 + term2 + 20 + np.e


# ======================================================
# INITIALIZE POPULATION
# ======================================================

def initialize_population(pop_size,
                          dimensions,
                          lower=-10,
                          upper=10):

    return np.random.uniform(
        lower,
        upper,
        (pop_size, dimensions)
    )


# ======================================================
# LEVY FUNCTION
# ======================================================

def levy(beta=1.5):

    sigma_u = (

        (
            math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
        )

        /

        (
            math.gamma((1 + beta)/2)
            * beta
            * (2 ** ((beta -1)/2))
        )

    ) ** (1/beta)

    u = np.random.normal(
        0,
        sigma_u
    )

    v = np.random.normal(
        0,
        1
    )

    step = u / (abs(v) ** (1/beta))

    return step


# ======================================================
# CLASSICAL RAO
# ======================================================

def rao1(objective_function,
         pop_size=50,
         dimensions=30,
         iterations=300):

    population = initialize_population(
        pop_size,
        dimensions
    )

    convergence = []

    for iteration in range(iterations):

        # ==========================================
        # FITNESS
        # ==========================================

        fitness = np.array(
            [objective_function(x)
             for x in population]
        )

        best = population[np.argmin(fitness)]

        worst = population[np.argmax(fitness)]

        new_population = []

        # ==========================================
        # UPDATE
        # ==========================================

        for x in population:

            r = np.random.rand()

            x_new = (
                x
                + r*(best - worst)
            )

            # Greedy Selection
            if objective_function(x_new) < objective_function(x):

                new_population.append(x_new)

            else:

                new_population.append(x)

        population = np.array(new_population)

        updated_fitness = np.array(
            [objective_function(x)
             for x in population]
        )

        convergence.append(
            np.min(updated_fitness)
        )

    return np.min(updated_fitness), convergence


# ======================================================
# LEVY GEOMETRIC QRAO
# ======================================================

def levy_geometric_qrao(objective_function,
                        pop_size=50,
                        dimensions=30,
                        iterations=300):

    population = initialize_population(
        pop_size,
        dimensions
    )

    convergence = []

    for iteration in range(iterations):

        # ==========================================
        # FITNESS
        # ==========================================

        fitness = np.array(
            [objective_function(x)
             for x in population]
        )

        best = population[np.argmin(fitness)]

        worst = population[np.argmax(fitness)]

        new_population = []

        # ==========================================
        # UPDATE
        # ==========================================

        for x in population:

            # Levy-based coefficients
            r1 = levy()

            r2 = levy()

            # Stability clipping
            r1 = np.tanh(r1)

            r2 = np.tanh(r2)

            # ======================================
            # NEW UPDATE EQUATION
            # ======================================

            x_new = (

                x

                + r1*(best - x)

                - r2*(worst - x)
            )

            # Greedy Selection
            if objective_function(x_new) < objective_function(x):

                new_population.append(x_new)

            else:

                new_population.append(x)

        population = np.array(new_population)

        updated_fitness = np.array(
            [objective_function(x)
             for x in population]
        )

        convergence.append(
            np.min(updated_fitness)
        )

    return np.min(updated_fitness), convergence


# ======================================================
# COMPARISON FUNCTION
# ======================================================

def compare_algorithms(objective_function,
                       function_name):

    runs = 20

    rao_results = []
    qrao_results = []

    rao_curve = None
    qrao_curve = None

    print(f"\n========== {function_name} ==========\n")

    for run in range(runs):

        # ======================================
        # CLASSICAL RAO
        # ======================================

        rao_best, rao_conv = rao1(

            objective_function=objective_function,

            pop_size=50,

            dimensions=30,

            iterations=300
        )

        rao_results.append(rao_best)

        # ======================================
        # NEW QRAO
        # ======================================

        qrao_best, qrao_conv = levy_geometric_qrao(

            objective_function=objective_function,

            pop_size=50,

            dimensions=30,

            iterations=300
        )

        qrao_results.append(qrao_best)

        # Save first convergence curves
        if run == 0:

            rao_curve = rao_conv
            qrao_curve = qrao_conv

    # ==========================================
    # PRINT RESULTS
    # ==========================================

    print("Classical RAO Mean Fitness:",
          np.mean(rao_results))

    print("Classical RAO Std:",
          np.std(rao_results))

    print()

    print("Levy-Geometric QRAO Mean Fitness:",
          np.mean(qrao_results))

    print("Levy-Geometric QRAO Std:",
          np.std(qrao_results))

    # ==========================================
    # PLOT
    # ==========================================

    plt.figure(figsize=(8,5))

    plt.plot(rao_curve,
             label="Classical RAO")

    plt.plot(qrao_curve,
             label="Levy-Geometric QRAO")

    plt.xlabel("Iterations")

    plt.ylabel("Best Fitness")

    plt.title(
        f"{function_name} Comparison"
    )

    plt.legend()

    plt.grid(True)

    plt.show()


# ======================================================
# RUN BENCHMARKS
# ======================================================

compare_algorithms(
    sphere,
    "Sphere Function"
)

compare_algorithms(
    rastrigin,
    "Rastrigin Function"
)

compare_algorithms(
    ackley,
    "Ackley Function"
)