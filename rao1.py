import random
import numpy as np
import math

def funct(x):
    return x**2 + 10*np.sin(5*x)

def rao1(iter, pop_size):

    pop = []

    for i in range(pop_size):
        pop.append(random.uniform(-10,10))

    fitness = []
    for i in range(pop_size):
        fitness.append(funct(pop[i]))

    best = min(fitness)
    worst = max(fitness)
    best_x = pop[np.argmin(fitness)]

    while iter>0 :
        for i in range(pop_size):
            pop[i] = pop[i] + random.uniform(0,1)*(best-worst)

            if(funct(pop[i])<best):
                best = funct(pop[i])
                best_x = pop[i]

            if(funct(pop[i])>worst):
                worst = funct(pop[i])

        iter = iter-1

    return best_x , best


iter = int(input("Enter number of iterations: "))
pop_size = int(input("Enter population size: "))

best_x , best_val = rao1(iter, pop_size)
print("Best x:", best_x)
print("Best value:", best_val)
            