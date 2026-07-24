import random
import numpy as np
import math

def funct(x):
    return x**2 + 10*np.sin(5*x)

def rao2(iter, pop_size):

    pop = []

    for i in range(pop_size):
        pop.append(random.uniform(-10,10))  

    while iter>0 :

        fitness = []
        for i in range(pop_size):
            fitness.append(funct(pop[i]))

        best = pop[np.argmin(fitness)]
        worst = pop[np.argmax(fitness)]

        best_val = min(fitness)
        worst_val = max(fitness)

        for i in range(pop_size):
            k = random.choice([idx for idx in range(pop_size) if idx != i])
            if fitness[i] < fitness[k]:
                new_x = pop[i] + random.uniform(0,1)*(best-worst) + random.uniform(0,1)*(abs(pop[i])-abs(pop[k]))
            else:
                new_x = pop[i] + random.uniform(0,1)*(best-worst) + random.uniform(0,1)*(abs(pop[k])-abs(pop[i]))
                
            if funct(new_x)<funct(pop[i]):
                pop[i] = new_x

        iter = iter-1

    fitness = [funct(x) for x in pop]

    best = pop[np.argmin(fitness)]
    best_val = min(fitness)

    return best , best_val


iter = int(input("Enter number of iterations: "))
pop_size = int(input("Enter population size: "))

best_x , best_val = rao2(iter, pop_size)
print("Best x:", best_x)
print("Best value:", best_val)
