import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. Problem Configuration & Randomized Seed Setup ---
np.random.seed(42)  
num_bits = 300       # 300-Dimension Combinatorial Knapsack

values = np.random.randint(10, 100, size=num_bits)
weights = np.random.randint(5, 50, size=num_bits)
capacity = int(0.35 * np.sum(weights))  

efficiency_ratios = values / weights
worst_to_best_indices = np.argsort(efficiency_ratios)

def vectorized_repair_and_evaluate(binary_matrix):
    repaired_matrix = binary_matrix.copy()
    total_weights = np.dot(repaired_matrix, weights)
    violators = total_weights > capacity
    
    if np.any(violators):
        for idx in worst_to_best_indices:
            still_violating = total_weights > capacity
            if not np.any(still_violating):
                break
            mask = still_violating & (repaired_matrix[:, idx] == 1)
            repaired_matrix[mask, idx] = 0
            total_weights[mask] -= weights[idx]
            
    fitness_scores = np.dot(repaired_matrix, values).astype(float)
    return repaired_matrix, fitness_scores


# --- 2. Vectorized Classical Continuous Rao-1 ---
class VectorizedClassicalRaoBinary:
    def __init__(self, pop_size, num_bits, max_iter):
        self.pop_size = pop_size
        self.num_bits = num_bits
        self.max_iter = max_iter
        self.positions = np.random.uniform(-3.0, 3.0, (pop_size, num_bits))
        
    def _apply_sigmoid(self):
        sigmoid = 1.0 / (1.0 + np.exp(-self.positions))
        return (np.random.rand(self.pop_size, self.num_bits) < sigmoid).astype(int)

    def optimize(self):
        fitness_history = []
        global_best_fitness = -float('inf')
        
        for iteration in range(self.max_iter):
            raw_binary_pop = self._apply_sigmoid()
            current_binary, fitness = vectorized_repair_and_evaluate(raw_binary_pop)
            
            best_idx = np.argmax(fitness)
            worst_idx = np.argmin(fitness)
            
            if fitness[best_idx] > global_best_fitness:
                global_best_fitness = fitness[best_idx]
            fitness_history.append(global_best_fitness)
            
            best_pos = self.positions[best_idx]
            worst_pos = self.positions[worst_idx]
            
            r = np.random.rand(self.pop_size, self.num_bits)
            self.positions += r * (best_pos - worst_pos)
            self.positions = np.clip(self.positions, -6.0, 6.0)
                
        return fitness_history


# --- 3. FIXED Vectorized Quantum-Inspired Rao ---
class FixedVectorizedQuantumRao:
    def __init__(self, pop_size, num_bits, max_iter, max_theta=0.07 * np.pi, min_theta=0.005 * np.pi):
        self.pop_size = pop_size
        self.num_bits = num_bits
        self.max_iter = max_iter
        self.max_theta = max_theta  
        self.min_theta = min_theta      
        self.q_pop = np.full((pop_size, num_bits), np.pi / 4.0)
        
    def _observe(self):
        prob_matrix = np.sin(self.q_pop) ** 2
        return (np.random.rand(self.pop_size, self.num_bits) < prob_matrix).astype(int)

    def optimize(self):
        fitness_history = []
        stagnation_counter = 0
        global_best_fitness = -float('inf')
        T_max = 60 # Slightly wider annealing cycle for larger dimensions
        
        for iteration in range(self.max_iter):
            T_cur = iteration % T_max
            current_delta_theta = self.min_theta + 0.5 * (self.max_theta - self.min_theta) * (1 + np.cos(np.pi * T_cur / T_max))
            
            raw_binary_pop = self._observe()
            current_binary, fitness = vectorized_repair_and_evaluate(raw_binary_pop)
            
            best_idx = np.argmax(fitness)
            worst_idx = np.argmin(fitness)
            
            if fitness[best_idx] > global_best_fitness:
                global_best_fitness = fitness[best_idx]
                stagnation_counter = 0  
            else:
                stagnation_counter += 1
                
            fitness_history.append(global_best_fitness)
            
            best_sol = current_binary[best_idx]
            worst_sol = current_binary[worst_idx]
            
            # --- FIXED PURE MATRIX GRADIENT UPDATE ---
            # Replicates the unvectorized logic by preserving full best-worst push
            base_gradient = best_sol - worst_sol  
            direction_matrix = np.tile(base_gradient, (self.pop_size, 1))
            
            self.q_pop += direction_matrix * current_delta_theta
            
            # Keep boundaries slightly wider (0.02) to preserve background mutations
            self.q_pop = np.clip(self.q_pop, 0.02, (np.pi / 2.0) - 0.02)
            
            # --- Enhanced Phase Shaking for 300 Dimensions ---
            if stagnation_counter >= 10: 
                # Stronger exploration noise to shatter the larger local trap
                noise = np.random.uniform(-0.08 * np.pi, 0.08 * np.pi, size=(self.pop_size, self.num_bits))
                noise[best_idx] = 0.0  
                self.q_pop += noise
                self.q_pop = np.clip(self.q_pop, 0.02, (np.pi / 2.0) - 0.02)
                stagnation_counter = 0
                    
        return fitness_history


# --- 4. EXACT GLOBAL OPTIMUM SOLVER ---
def solve_knapsack_exactly(W, w, v, n):
    dp = np.zeros(W + 1)
    for i in range(n):
        for weight in range(W, w[i] - 1, -1):
            dp[weight] = max(dp[weight], dp[weight - w[i]] + v[i])
    return dp[W]

print("Calculating true global max baseline...")
true_global_max = solve_knapsack_exactly(capacity, weights, values, num_bits)


# --- 5. Execution Routine ---
POPULATION_SIZE = 30
MAX_ITERATIONS = 300

start_classical = time.time()
classical_solver = VectorizedClassicalRaoBinary(POPULATION_SIZE, num_bits, MAX_ITERATIONS)
classical_perf = classical_solver.optimize()
classical_time = time.time() - start_classical

start_quantum = time.time()
quantum_solver = FixedVectorizedQuantumRao(POPULATION_SIZE, num_bits, MAX_ITERATIONS)
quantum_perf = quantum_solver.optimize()
quantum_time = time.time() - start_quantum


# --- 6. Print Profile Results ---
print("\n" + "="*70)
print(f"  RE-ENGINEERED HEAD-TO-HEAD PROFILE (True Max: {true_global_max:.0f})")
print("="*70)
print(f"Parameter                | Classical Rao (Sigmoid)  | Fixed Vectorized Quantum Rao")
print("-"*70)
print(f"Score Reached            | {classical_perf[-1]:<24.1f} | {quantum_perf[-1]:.1f}")
print(f"Accuracy (%)             | {(classical_perf[-1]/true_global_max)*100:<24.2f}% | {(quantum_perf[-1]/true_global_max)*100:.2f}%")
print(f"Speed (sec)              | {classical_time:<24.4f} | {quantum_time:.4f}")
print("="*70)

# --- 7. Plotting ---
plt.figure(figsize=(11, 6))
plt.plot(classical_perf, label="Classical Continuous Rao", color="crimson", linewidth=2)
plt.plot(quantum_perf, label="Fixed Vectorized Quantum Rao (Corrected Matrix)", color="darkviolet", linewidth=2.5)
plt.axhline(y=true_global_max, color="black", linestyle="--", alpha=0.7, label=f"True Global Max ({true_global_max:.0f})")
plt.title("Corrected Vectorized Convergence Test: 300-Dimension Combinatorial Knapsack", fontsize=12, fontweight="bold")
plt.xlabel("Iteration Step Count")
plt.ylabel("Valid Found Value")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
