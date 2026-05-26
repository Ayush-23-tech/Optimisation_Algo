import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import wilcoxon

# --- 1. PROBLEM SETUP: LARGE-SCALE QUADRATIC ASSIGNMENT PROBLEM (QAP) ---
np.random.seed(42)
N = 25  # Number of facilities and locations (Search space = 25! permutations)
NUM_BITS = N * 8  # 8 bits assigned per facility to create a continuous-like probability grid

# Generate random Flow and Distance matrices (Symmetric layout)
flow_matrix = np.random.randint(5, 50, size=(N, N))
np.fill_diagonal(flow_matrix, 0)
flow_matrix = (flow_matrix + flow_matrix.T) // 2

distance_matrix = np.random.randint(10, 100, size=(N, N))
np.fill_diagonal(distance_matrix, 0)
distance_matrix = (distance_matrix + distance_matrix.T) // 2

def qap_evaluate(binary_matrix):
    """
    IEEE Compliant Permutation Decoder & Evaluation Function.
    Decodes the raw bit matrix into a valid, unique facility permutation sequence.
    """
    pop_size = binary_matrix.shape[0]
    
    # Deconstruct the flat bit matrix into N segments per individual
    reshaped = binary_matrix.reshape(pop_size, N, 8)
    bit_powers = 2 ** np.arange(8)[::-1]
    
    # Convert bits to unique decimal ranks to handle the permutation mapping
    scores = np.dot(reshaped, bit_powers)
    permutations = np.argsort(scores, axis=1) # Guarantees unique assignments (No duplicates)
    
    fitness_scores = np.zeros(pop_size)
    for i in range(pop_size):
        p = permutations[i]
        # Core QAP Math: Sum(Flow[i,j] * Distance[p[i], p[j]])
        # Efficient vector trace operation to calculate total layout cost
        fitness_scores[i] = np.trace(np.dot(flow_matrix, distance_matrix[p][:, p]))
        
    return fitness_scores


# --- 2. THE COMPETITOR: GOLD-STANDARD BINARY PSO (BPSO) ---
class VectorizedBinaryPSO:
    def __init__(self, pop_size, num_bits, max_iter):
        self.pop_size = pop_size
        self.num_bits = num_bits
        self.max_iter = max_iter
        self.positions = (np.random.rand(pop_size, num_bits) > 0.5).astype(int)
        self.velocities = np.zeros((pop_size, num_bits))
        
        self.pbest = self.positions.copy()
        self.pbest_fit = qap_evaluate(self.pbest)
        
    def optimize(self):
        convergence = []
        w, c1, c2 = 0.7, 1.4, 1.4  # Standard optimized BPSO coefficients
        
        for iteration in range(self.max_iter):
            gbest_idx = np.argmin(self.pbest_fit)
            gbest = self.pbest[gbest_idx]
            
            # Update discrete swarm velocities
            r1, r2 = np.random.rand(self.pop_size, self.num_bits), np.random.rand(self.pop_size, self.num_bits)
            self.velocities = (w * self.velocities + 
                               c1 * r1 * (self.pbest - self.positions) + 
                               c2 * r2 * (gbest - self.positions))
            self.velocities = np.clip(self.velocities, -4.0, 4.0)
            
            # Pass velocities through Sigmoid Transfer Function
            sigmoid = 1.0 / (1.0 + np.exp(-self.velocities))
            self.positions = (np.random.rand(self.pop_size, self.num_bits) < sigmoid).astype(int)
            
            # Evaluate new population configurations
            fitness = qap_evaluate(self.positions)
            improved = fitness < self.pbest_fit
            self.pbest[improved] = self.positions[improved]
            self.pbest_fit[improved] = fitness[improved]
            
            convergence.append(np.min(self.pbest_fit))
            
        return convergence


# --- 3. THE PROPOSED CHAMPION: VECTORIZED QUANTUM RAO (Q-RAO) ---
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
        convergence = []
        stagnation_counter = 0
        global_best_fitness = float('inf')
        T_max = 50 
        
        for iteration in range(self.max_iter):
            T_cur = iteration % T_max
            current_delta_theta = self.min_theta + 0.5 * (self.max_theta - self.min_theta) * (1 + np.cos(np.pi * T_cur / T_max))
            
            raw_binary_pop = self._observe()
            fitness = qap_evaluate(raw_binary_pop)
            
            best_idx = np.argmin(fitness)
            worst_idx = np.argmax(fitness)
            
            if fitness[best_idx] < global_best_fitness:
                global_best_fitness = fitness[best_idx]
                stagnation_counter = 0  
            else:
                stagnation_counter += 1
                
            convergence.append(global_best_fitness)
            
            best_sol = raw_binary_pop[best_idx]
            worst_sol = raw_binary_pop[worst_idx]
            
            # Vectorized Matrix Quantum Update Step
            base_gradient = best_sol - worst_sol  
            direction_matrix = np.tile(base_gradient, (self.pop_size, 1))
            
            self.q_pop += direction_matrix * current_delta_theta
            self.q_pop = np.clip(self.q_pop, 0.02, (np.pi / 2.0) - 0.02)
            
            # Quantum Phase-Shaking (Decoherence) to destroy local traps
            if stagnation_counter >= 12:
                noise = np.random.uniform(-0.08 * np.pi, 0.08 * np.pi, size=(self.pop_size, self.num_bits))
                noise[best_idx] = 0.0  
                self.q_pop += noise
                self.q_pop = np.clip(self.q_pop, 0.02, (np.pi / 2.0) - 0.02)
                stagnation_counter = 0
                    
        return convergence


# --- 4. IEEE COMPLIANT MULTI-RUN PROFILER EXECUTION ---
RUNS = 15
POP_SIZE = 30
MAX_ITER = 200

bpso_finals, qrao_finals = [], []
bpso_curve, qrao_curve = None, None

print(f"Launching IEEE Discrete Benchmark Suite across {RUNS} active runs...")

# Benchmark Binary PSO Baseline
start = time.time()
for r in range(RUNS):
    solver = VectorizedBinaryPSO(POP_SIZE, NUM_BITS, MAX_ITER)
    curve = solver.optimize()
    bpso_finals.append(curve[-1])
    if r == 0: bpso_curve = np.array(curve)
bpso_time = (time.time() - start) / RUNS

# Benchmark Proposed Vectorized Quantum Rao
start = time.time()
for r in range(RUNS):
    solver = FixedVectorizedQuantumRao(POP_SIZE, NUM_BITS, MAX_ITER)
    curve = solver.optimize()
    qrao_finals.append(curve[-1])
    if r == 0: qrao_curve = np.array(curve)
qrao_time = (time.time() - start) / RUNS

# Non-Parametric Wilcoxon Statistical Validation Check
try:
    _, p_value = wilcoxon(bpso_finals, qrao_finals)
except ValueError:
    p_value = 1.0

# --- 5. PRINT OFFICIAL METRICS REPORT ---
print("\n" + "="*80)
print(f" IEEE COMBINATORIAL REPORT: COMPLEX QUADRATIC ASSIGNMENT PROBLEM (QAP)")
print("="*80)
print(f"Statistical Metric        | Binary PSO Baseline      | Your Vectorized Q-Rao")
print("-"*80)
print(f"Best Layout Cost (MIN)    | {np.min(bpso_finals):<24.2f} | {np.min(qrao_finals):.2f}")
print(f"Worst Layout Cost (MAX)   | {np.max(bpso_finals):<24.2f} | {np.max(qrao_finals):.2f}")
print(f"Mean Cost Performance     | {np.mean(bpso_finals):<24.2f} | {np.mean(qrao_finals):.2f}")
print(f"Standard Deviation (Std)  | {np.std(bpso_finals):<24.2f} | {np.std(qrao_finals):.2f}")
print(f"Avg Computation Speed    | {bpso_time:<24.4f} sec | {qrao_time:.4f} sec")
print(f"Wilcoxon p-value Result   | p-value = {p_value:.4e}   | (Alpha Limit = 0.05)")
print(f"IEEE Statistical Winner   | {'Q-Rao Wins' if np.mean(qrao_finals) < np.mean(bpso_finals) else 'BPSO Baseline Wins' if np.mean(bpso_finals) < np.mean(qrao_finals) else 'Draw'}")
print("="*80)

# --- 6. PLOT CONVERGENCE PROGRESS ---
plt.figure(figsize=(10, 5.5))
plt.plot(bpso_curve, label="Binary PSO (Gold Standard Baseline)", color="crimson", linewidth=1.8)
plt.plot(qrao_curve, label="Proposed Vectorized Quantum Rao", color="darkviolet", linewidth=2.3)
plt.title("IEEE Combinatorial Benchmark Profile: 25-Facility QAP Matrix", fontsize=12, fontweight="bold")
plt.xlabel("Generation Evaluation Iterations", fontsize=11)
plt.ylabel("Minimum Found Layout Cost Matrix (Lower is Better)", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
