import random
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Objective Function
# =========================

def funct(x):
    return x**2 + 10*np.sin(5*x)


# =========================
# Rao-1 Algorithm
# =========================

def rao1(iterations, pop_size):

    # Initialize population
    pop = []

    for i in range(pop_size):
        pop.append(random.uniform(-10,10))

    # To track convergence
    convergence = []

    # To track diversity
    diversity_history = []

    # Main optimization loop
    while iterations > 0:

        # Fitness calculation
        fitness = []

        for i in range(pop_size):
            fitness.append(funct(pop[i]))

        # Best and worst particles
        best = pop[np.argmin(fitness)]
        worst = pop[np.argmax(fitness)]

        # =========================
        # Rao-1 Update
        # =========================

        for i in range(pop_size):

            new_x = (
                pop[i]
                + random.uniform(0,1)*(best - worst)
            )

            # Greedy selection
            if funct(new_x) < funct(pop[i]):
                pop[i] = new_x

        # =========================
        # Track convergence
        # =========================

        fitness = [funct(x) for x in pop]

        best_val = min(fitness)

        convergence.append(best_val)

        # =========================
        # Track diversity
        # =========================

        diversity = np.mean(
            np.abs(pop - np.mean(pop))
        )

        diversity_history.append(diversity)

        iterations -= 1

    # Final best particle
    fitness = [funct(x) for x in pop]

    best = pop[np.argmin(fitness)]
    best_val = min(fitness)

    return best, best_val, convergence, diversity_history


# =========================
# User Input
# =========================

iterations = int(input("Enter number of iterations: "))
pop_size = int(input("Enter population size: "))


# =========================
# Run Rao-1
# =========================

best_x, best_val, convergence, diversity = rao1(
    iterations,
    pop_size
)


# =========================
# Final Result
# =========================

print("\nBest x:", best_x)
print("Best value:", best_val)


# =========================
# Convergence Plot
# =========================

plt.figure(figsize=(8,5))

plt.plot(convergence)

plt.xlabel("Iterations")
plt.ylabel("Best Fitness")
plt.title("Rao-1 Convergence Curve")

plt.grid(True)

plt.show()


# =========================
# Diversity Plot
# =========================

plt.figure(figsize=(8,5))

plt.plot(diversity)

plt.xlabel("Iterations")
plt.ylabel("Population Diversity")
plt.title("Rao-1 Diversity Curve")

plt.grid(True)

plt.show()