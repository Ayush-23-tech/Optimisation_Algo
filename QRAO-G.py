import numpy as np

def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    n = len(x)
    return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))

def ackley(x):
    n = len(x)

    term1 = -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/n))
    term2 = -np.exp(np.sum(np.cos(2*np.pi*x))/n)

    return term1 + term2 + 20 + np.e


def initialize_population(pop_size, dimensions, lower=-10, upper=10):

    return np.random.uniform(lower, upper,
                             (pop_size, dimensions))


def qrao(objective_function,
         pop_size=30,
         dimensions=2,
         iterations=100):

    population = initialize_population(pop_size,
                                       dimensions)

    convergence = []
    diversity_history = []

    for iteration in range(iterations):

        fitness = np.array(
            [objective_function(x)
             for x in population]
        )

        best_idx = np.argmin(fitness)
        worst_idx = np.argmax(fitness)

        best = population[best_idx]
        worst = population[worst_idx]

        new_population = []

        for x in population:

            r1 = np.random.rand()
            r2 = np.random.rand()

            # Attractor
            A = x + r1*(best - x) - r2*(worst - x)

            # Distance-based sigma
            sigma = 0.5 * np.linalg.norm(A - x)

            # Gaussian localization
            x_new = np.random.normal(A, sigma)

            # Greedy selection
            if objective_function(x_new) < objective_function(x):
                new_population.append(x_new)
            else:
                new_population.append(x)

        population = np.array(new_population)

        # Track convergence
        best_fitness = np.min(fitness)
        convergence.append(best_fitness)

        # Track diversity
        diversity = np.mean(
            np.linalg.norm(population -
                           np.mean(population, axis=0),
                           axis=1)
        )

        diversity_history.append(diversity)

    return population, convergence, diversity_history


population, convergence, diversity = qrao(
    rastrigin,
    pop_size=50,
    dimensions=10,
    iterations=200
)

import matplotlib.pyplot as plt

# Convergence plot
plt.figure(figsize=(8,5))
plt.plot(convergence)
plt.xlabel("Iterations")
plt.ylabel("Best Fitness")
plt.title("Convergence Curve")
plt.grid(True)
plt.show()

# Diversity plot
plt.figure(figsize=(8,5))
plt.plot(diversity)
plt.xlabel("Iterations")
plt.ylabel("Diversity")
plt.title("Diversity Curve")
plt.grid(True)
plt.show()
