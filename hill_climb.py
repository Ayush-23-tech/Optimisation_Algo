def funct(x):
    return x**2 + 3*x + 5

def hill_climb(x, step_size):

    while True :
        current_val = funct(x)

        left = funct(x-step_size)
        print(f"Left neighbour value: {left}")
        right = funct(x+step_size) 
        print(f"Right neighbour value: {right}")

        best_neighbour = min(left, right)
        print(f"Best neighbour value: {best_neighbour}")
    
        if (best_neighbour< current_val ) :
            if(left < right) :
                 x= x-step_size

            else :
                x = x+step_size

        else :
            break

        print(f"Current x: {x}, Current value: {funct(x)}")

    return x, funct(x)


initial_x = float(input("Enter initial x: "))
step_size = float(input("Enter step size: "))

best_x, best_value = hill_climb(initial_x, step_size)

print("Best x:", best_x)
print("Minimum value:", best_value)


