import random
import math 

def funct(x):
    return x**2 + 10*math.sin(x) + 1

def simulated_annealing(x):
    temp = 100
    cool_factor = 0.85


    while temp > 0.01 :

        new_x = x + random.randint(-1,1)

        current_val = funct(x)

        new_val = funct(new_x)

        delta = funct(new_x) - funct(x)
        acceptance_prob = math.exp(-delta/temp)

        if delta < 0:
            x = new_x

        else:
            acceptance_prob = math.exp(-delta/temp)

            if random.random() < acceptance_prob:
                    x = new_x
                    temp = temp * cool_factor

    return x , funct(x)


initial_x = float(input("Enter initial x: "))
print("best_x : " , simulated_annealing(initial_x)[0])
print("Minimum value : " , simulated_annealing(initial_x)[1])

