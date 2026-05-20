import math
import random
import numpy as np

def funct(candidate) :
    total = 0

    for x in candidate:
        total = total + x**2

    return total


def rao_nd(dimension,pop_size, iter):
    population = []

    for i in range(pop_size):
        candidate = []
   

        for x in range(dimension):
            candidate.append(random.uniform(-10,10))

        
        candidate = np.array(candidate)
        population.append(candidate)

    while iter>0 :
        fitness = []
        for i in range(pop_size):
            fitness.append(funct(candidate))

        best = population[np.argmin(fitness)]
        worst = population[np.argmax(fitness)]

        best_val = min(fitness)
        worst_val = max(fitness)

        for i in range(pop_size):
            new_x = population[i] + random.uniform(0,1)*(best - worst)
            if(funct(new_x)<funct(population[i])):
                population[i] = new_x

        iter = iter-1

    fitness = [funct(x) for x in population]
    best = population[np.argmin(fitness)]
    best_val = min(fitness)

    return best , best_val


pop_size = int(input("enter the population size : "))
iter = int(input("enter maximum no of iterations : "))
dimen = int(input("enter the dimension of each candidate : "))

best_x , best_val = rao_nd(dimen , pop_size , iter)
print("best value of x is : ", best_x)
print("best value of function is : ", best_val)