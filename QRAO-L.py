import numpy as np
import matplotlib.pyplot as plt
import math


# =========================================
# Benchmark Functions
# =========================================

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


# =========================================
# Population Initialization
# =========================================

def initialize_population(pop_size,
                          dimensions,
                          lower=-10,
                          upper=10):

    return np.random.uniform(
        lower,
        upper,
        (pop_size, dimensions)
    )


# =========================================
# Levy Flight Generator
# =========================================

def levy(dimensions, beta=1.5):

    # Mantegna Algorithm

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


# =========================================
# QRAO-L Algorithm
# =========================================

def qrao_levy(objective_function,
              pop_size=50,
              dimensions=30,
              iterations=300):

    # Initialize population
    population = initialize_population(
        pop_size,
        dimensions
    )

    convergence = []
    diversity_history = []

    # =====================================
    # Main Optimization Loop
    # =====================================

    for iteration in range(iterations):

        # Fitness evaluation
        fitness = np.array(
            [objective_function(x)
             for x in population]
        )

        # Best and worst particles
        best_idx = np.argmin(fitness)
        worst_idx = np.argmax(fitness)

        best = population[best_idx]
        worst = population[worst_idx]

        new_population = []

        # =================================
        # Particle Update
        # =================================

        for x in population:

            # Random attraction/repulsion
            r1 = np.random.rand()
            r2 = np.random.rand()

            # =================================
            # Attractor
            # =================================

            A = (
                x
                + r1*(best - x)
                - r2*(worst - x)
            )

            # =================================
            # Adaptive Sigma Decay
            # =================================

            sigma = (
                0.5
                * (1 - iteration/iterations)
                * np.linalg.norm(A - x)
            )

            sigma = max(sigma, 1e-6)

            # =================================
            # Levy Localization
            # =================================

            levy_step = levy(dimensions)

            x_new = (
                A
                + sigma * levy_step
            )

            # =================================
            # Greedy Selection
            # =================================

            if objective_function(x_new) < objective_function(x):

                new_population.append(x_new)

            else:

                new_population.append(x)

        # Update population
        population = np.array(new_population)

        # =====================================
        # Updated Fitness
        # =====================================

        updated_fitness = np.array(
            [objective_function(x)
             for x in population]
        )

        best_fitness = np.min(updated_fitness)

        convergence.append(best_fitness)

        # =====================================
        # Diversity Tracking
        # =====================================

        diversity = np.mean(

            np.linalg.norm(

                population
                - np.mean(population, axis=0),

                axis=1
            )
        )

        diversity_history.append(diversity)

    return (
        best_fitness,
        convergence,
        diversity_history
    )


# =========================================
# RUN QRAO-L
# =========================================

best_fitness, convergence, diversity = qrao_levy(

    objective_function=rastrigin,

    pop_size=50,

    dimensions=30,

    iterations=300
)


# =========================================
# FINAL RESULT
# =========================================

print("\nBest Fitness:\n")
print(best_fitness)


# =========================================
# Convergence Plot
# =========================================

plt.figure(figsize=(8,5))

plt.plot(convergence)

plt.xlabel("Iterations")
plt.ylabel("Best Fitness")

plt.title("QRAO-L Convergence Curve")

plt.grid(True)

plt.show()


# =========================================
# Diversity Plot
# =========================================

plt.figure(figsize=(8,5))

plt.plot(diversity)

plt.xlabel("Iterations")
plt.ylabel("Population Diversity")

plt.title("QRAO-L Diversity Curve")

plt.grid(True)

plt.show()