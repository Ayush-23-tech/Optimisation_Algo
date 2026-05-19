import numpy as np
import math
import random

def funct(x):
    return x**2 + 10*np.sin(5*x)

def rao3(iters , pop_size):
    pop = []

    for i in range(pop_size):
        pop.append(random.uniform(-10,10))

    while iters > 0 :

        fitness = []
        for i in range(pop_size):
            fitness.append(funct(pop[i]))

        best  = pop[np.argmin(fitness)]
        worst = pop[np.argmax(fitness)]

        best_val = min(fitness)
        worst_val = max(fitness)

        for i in range(pop_size):
            x1, x2 = random.sample(pop,2)
            new_x = pop[i] + random.uniform(0,1)*(best-worst) + random.uniform(0,1)*(abs(x1)-x2)

            if(funct(new_x)<funct(pop[i])):
                pop[i]= new_x

        iters = iters - 1 

    fitness = [funct(x) for x in pop]

    best = pop[np.argmin(fitness)]
    best_val = min(fitness)

    return best , best_val


iters = int(input("enter maxm no of iterations : "))
pop_size = int(input("enter no of candidates : "))

best_x , best_val = rao3(iters, pop_size)

print("the best value of x is : ",best_x)
print("the minimum value of function is : ",best_val)
            

