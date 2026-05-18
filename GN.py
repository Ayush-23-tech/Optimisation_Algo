import math 
import random
import numpy as np

def funct(x):
    return x**2 + 3*x + 6

def genetic_algo(population_size ,  generations):

    population = []

    for i in range(population_size):
        population.append(random.uniform(-10,10))

    
    while generations > 0:
        fitness = []
        for i in range(population_size):
            fitness.append(funct(population[i]))

        probabilities = [0] * population_size

        for i in range(population_size):
            score = 1/(1+fitness[i])
            probabilities[i] = score

        sum_prob = np.sum(probabilities)

        for i in range(population_size):
            probabilities[i] = probabilities[i]/sum_prob

        new_population = []

        while len(new_population) < population_size:
            parents = random.choices(population, weights=probabilities, k=2)

            child = (parents[0] + parents[1]) / 2
            child = child + random.uniform(-1,1)
            new_population.append(child)
        population = new_population
        generations -= 1

    fitness = []

    for i in range(population_size):
        fitness.append(funct(population[i]))

    best_index = np.argmin(fitness)

    return population[best_index], fitness[best_index]  

population_size = int(input("Enter population size: "))
generations = int(input("Enter number of generations: "))
best_x, best_value = genetic_algo(population_size, generations)
print("Best x:", best_x)
print("Minimum value:", best_value)


    

    

