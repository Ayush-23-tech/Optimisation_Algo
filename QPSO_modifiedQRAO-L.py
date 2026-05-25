import numpy as np
import matplotlib.pyplot as plt
import math


# ======================================================
# BENCHMARK FUNCTIONS
# ======================================================

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
# POPULATION INITIALIZATION
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
# LEVY FLIGHT
# ======================================================

def levy(dimensions, beta=1.5):

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
        sigma_u,
        dimensions
    )

    v = np.random.normal(
        0,
        1,
        dimensions
    )

    step = u / (np.abs(v) ** (1/beta))

    return step


# ======================================================
# NEW QRAO-L WITH MBEST CONTRACTION
# ======================================================

def qrao_mbest(objective_function,
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

        # Best and worst particles
        best_idx = np.argmin(fitness)
        worst_idx = np.argmax(fitness)

        best = population[best_idx]
        worst = population[worst_idx]

        # ==========================================
        # MBEST
        # ==========================================

        mbest = np.mean(population, axis=0)

        new_population = []

        # ==========================================
        # ADAPTIVE LAMBDA
        # ==========================================

        # stronger contraction later
        lam = iteration / iterations

        # ==========================================
        # UPDATE PARTICLES
        # ==========================================

        for x in population:

            r1 = np.random.rand()
            r2 = np.random.rand()

            # ======================================
            # NEW ATTRACTOR
            # ======================================

            A = (

                x

                + r1*(best - x)

                - r2*(worst - x)

                + lam*(mbest - x)
            )

            # ======================================
            # ADAPTIVE SIGMA DECAY
            # ======================================

            sigma = (

                0.5

                * (1 - iteration/iterations)

                * np.linalg.norm(A - x)
            )

            sigma = max(sigma, 1e-6)

            # ======================================
            # LEVY LOCALIZATION
            # ======================================

            levy_step = levy(dimensions)

            x_new = (
                A
                + sigma * levy_step
            )

            # ======================================
            # GREEDY SELECTION
            # ======================================

            if objective_function(x_new) < objective_function(x):

                new_population.append(x_new)

            else:

                new_population.append(x)

        population = np.array(new_population)

        # ==========================================
        # UPDATED FITNESS
        # ==========================================

        updated_fitness = np.array(
            [objective_function(x)
             for x in population]
        )

        convergence.append(
            np.min(updated_fitness)
        )

    return np.min(updated_fitness), convergence


# ======================================================
# QPSO
# ======================================================

def qpso(objective_function,
          pop_size=50,
          dimensions=30,
          iterations=300):

    population = initialize_population(
        pop_size,
        dimensions
    )

    pbest = population.copy()

    pbest_fitness = np.array(
        [objective_function(x)
         for x in pbest]
    )

    convergence = []

    beta = 0.75

    for iteration in range(iterations):

        # ==========================================
        # GLOBAL BEST
        # ==========================================

        gbest_idx = np.argmin(pbest_fitness)

        gbest = pbest[gbest_idx]

        # ==========================================
        # MBEST
        # ==========================================

        mbest = np.mean(pbest, axis=0)

        new_population = []

        for i in range(pop_size):

            x = population[i]

            phi = np.random.rand()

            # ======================================
            # ATTRACTOR
            # ======================================

            p = (
                phi * pbest[i]
                + (1 - phi) * gbest
            )

            u = np.random.rand()

            sign = 1 if np.random.rand() < 0.5 else -1

            # ======================================
            # QPSO UPDATE
            # ======================================

            x_new = (

                p

                + sign

                * beta

                * np.abs(mbest - x)

                * np.log(1/u)
            )

            new_population.append(x_new)

        population = np.array(new_population)

        # ==========================================
        # UPDATE PERSONAL BESTS
        # ==========================================

        for i in range(pop_size):

            fit = objective_function(population[i])

            if fit < pbest_fitness[i]:

                pbest[i] = population[i]

                pbest_fitness[i] = fit

        convergence.append(
            np.min(pbest_fitness)
        )

    return np.min(pbest_fitness), convergence


# ======================================================
# COMPARISON FUNCTION
# ======================================================

def compare_algorithms(objective_function,
                       function_name):

    runs = 20

    qrao_results = []
    qpso_results = []

    qrao_curve = None
    qpso_curve = None

    print(f"\n========== {function_name} ==========\n")

    for run in range(runs):

        # ======================================
        # NEW QRAO
        # ======================================

        qrao_best, qrao_conv = qrao_mbest(

            objective_function=objective_function,

            pop_size=50,

            dimensions=30,

            iterations=300
        )

        qrao_results.append(qrao_best)

        # ======================================
        # QPSO
        # ======================================

        qpso_best, qpso_conv = qpso(

            objective_function=objective_function,

            pop_size=50,

            dimensions=30,

            iterations=300
        )

        qpso_results.append(qpso_best)

        # save first convergence curve
        if run == 0:

            qrao_curve = qrao_conv
            qpso_curve = qpso_conv

    # ==========================================
    # RESULTS
    # ==========================================

    print("NEW QRAO Mean Fitness:",
          np.mean(qrao_results))

    print("NEW QRAO Std:",
          np.std(qrao_results))

    print()

    print("QPSO Mean Fitness:",
          np.mean(qpso_results))

    print("QPSO Std:",
          np.std(qpso_results))

    # ==========================================
    # CONVERGENCE PLOT
    # ==========================================

    plt.figure(figsize=(8,5))

    plt.plot(qrao_curve,
             label="NEW QRAO")

    plt.plot(qpso_curve,
             label="QPSO")

    plt.xlabel("Iterations")

    plt.ylabel("Best Fitness")

    plt.title(
        f"{function_name} Convergence Comparison"
    )

    plt.legend()

    plt.grid(True)

    plt.show()


# ======================================================
# RUN COMPARISONS
# ======================================================

compare_algorithms(
    rastrigin,
    "Rastrigin Function"
)

compare_algorithms(
    ackley,
    "Ackley Function"
)