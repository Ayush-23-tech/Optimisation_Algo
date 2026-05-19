import math
import numpy as np
import random

def funct(x):
    return x**2 + 10*np.sin(x) + 10

def pso(num_cand , iter):

    candidates = []
    for i in range(num_cand):
        candidates.append(random.uniform(-10, 10))

    velocities = []
    for i in range(num_cand):
        velocities.append(random.uniform(-1, 1))

    pbest = []
    pbest = candidates.copy()

    fitness = []
    for i in range(num_cand):
        fitness.append(funct(candidates[i]))

    gbest = min(fitness)

    w=0.8
    c1 = 2.5
    c2 = 2.5

    while(iter>0):
        for i in range(num_cand):
            velocities[i] = velocities[i]*w + c1*(random.uniform(0, 1))*(pbest[i]-candidates[i]) + c2*(random.uniform(0, 1))*(gbest - candidates[i])
            candidates[i] = candidates[i] + velocities[i]

            if(candidates[i]<pbest[i]):
                pbest[i] = candidates[i]

            if(funct(candidates[i])<funct(gbest)):
                gbest = candidates[i]

        iter -= 1

    return gbest , funct(gbest)


num_cand = int(input("enter no of candidates : "))
iter = int(input("enter maximum no of iterations : "))

best_x , best_val = pso(num_cand , iter)
print("best value of x is ", best_x)
print("best function value is ", best_val)


    
