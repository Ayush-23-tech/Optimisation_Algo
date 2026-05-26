import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import wilcoxon

# ======================================================
# SCALABLE QAP EVALUATION ENGINE
# ======================================================
def generate_qap_matrices(N_size):
    """Generates synthetic symmetric Flow and Distance matrices for a given size N."""
    np.random.seed(100 + N_size) # Stable problem generation per size
    flow = np.random.randint(5, 50, size=(N_size, N_size))
    np.fill_diagonal(flow, 0)
    flow = (flow + flow.T) // 2

    dist = np.random.randint(10, 100, size=(N_size, N_size))
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) // 2
    return flow, dist

def qap_evaluate_scaled(binary_matrix, N_size, flow, dist):
    pop_size = binary_matrix.shape[0]
    reshaped = binary_matrix.reshape(pop_size, N_size, 8)
    bit_powers = 2 ** np.arange(8)[::-1]
    
    scores = np.dot(reshaped, bit_powers)
    permutations = np.argsort(scores, axis=1)
    
    fitness_scores = np.zeros(pop_size)
    for i in range(pop_size):
        p = permutations[i]
        fitness_scores[i] = np.trace(np.dot(flow, dist[p][:, p]))
    return fitness_scores

# ======================================================
# ALGORITHM IMPLEMENTATIONS
# ======================================================
class VectorizedBinaryPSO:
    def __init__(self, pop_size, num_bits, max_iter, N_size, flow, dist):
        self.pop_size = pop_size
        self.num_bits = num_bits
        self.max_iter = max_iter
        self.N_size = N_size
        self.flow = flow
        self.dist = dist
        self.positions = (np.random.rand(pop_size, num_bits) > 0.5).astype(int)
        self.velocities = np.zeros((pop_size, num_bits))
        self.pbest = self.positions.copy()
        self.pbest_fit = qap_evaluate_scaled(self.pbest, self.N_size, self.flow, self.dist)
        
    def optimize(self):
        w, c1, c2 = 0.7, 1.4, 1.4
        for _ in range(self.max_iter):
            gbest = self.pbest[np.argmin(self.pbest_fit)]
            r1, r2 = np.random.rand(self.pop_size, self.num_bits), np.random.rand(self.pop_size, self.num_bits)
            self.velocities = (w * self.velocities + c1 * r1 * (self.pbest - self.positions) + c2 * r2 * (gbest - self.positions))
            self.velocities = np.clip(self.velocities, -4.0, 4.0)
            sigmoid = 1.0 / (1.0 + np.exp(-self.velocities))
            self.positions = (np.random.rand(self.pop_size, self.num_bits) < sigmoid).astype(int)
            
            fitness = qap_evaluate_scaled(self.positions, self.N_size, self.flow, self.dist)
            improved = fitness < self.pbest_fit
            self.pbest[improved] = self.positions[improved]
            self.pbest_fit[improved] = fitness[improved]
        return np.min(self.pbest_fit)

class FixedVectorizedQuantumRao:
    def __init__(self, pop_size, num_bits, max_iter, N_size, flow, dist):
        self.pop_size = pop_size
        self.num_bits = num_bits
        self.max_iter = max_iter
        self.N_size = N_size
        self.flow = flow
        self.dist = dist
        self.q_pop = np.full((pop_size, num_bits), np.pi / 4.0)
        
    def _observe(self):
        return (np.random.rand(self.pop_size, self.num_bits) < np.sin(self.q_pop)**2).astype(int)

    def optimize(self):
        stagnation_counter = 0
        global_best_fitness = float('inf')
        
        for iteration in range(self.max_iter):
            T_cur = iteration % 50
            current_delta_theta = 0.005*np.pi + 0.5*(0.07*np.pi - 0.005*np.pi)*(1 + np.cos(np.pi * T_cur / 50))
            
            raw_binary_pop = self._observe()
            fitness = qap_evaluate_scaled(raw_binary_pop, self.N_size, self.flow, self.dist)
            best_idx, worst_idx = np.argmin(fitness), np.argmax(fitness)
            
            if fitness[best_idx] < global_best_fitness:
                global_best_fitness = fitness[best_idx]
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            self.q_pop += np.tile(raw_binary_pop[best_idx] - raw_binary_pop[worst_idx], (self.pop_size, 1)) * current_delta_theta
            self.q_pop = np.clip(self.q_pop, 0.02, (np.pi / 2.0) - 0.02)
            
            if stagnation_counter >= 12:
                noise = np.random.uniform(-0.08 * np.pi, 0.08 * np.pi, size=(self.pop_size, self.num_bits))
                noise[best_idx] = 0.0
                self.q_pop += noise
                self.q_pop = np.clip(self.q_pop, 0.02, (np.pi / 2.0) - 0.02)
                stagnation_counter = 0
        return global_best_fitness

# ======================================================
# SCALABILITY EXECUTION FRAMEWORK
# ======================================================
TEST_SIZES = [12, 24, 40] # Small, Medium, Large Scale instances
RUNS = 10
POP_SIZE = 30
MAX_ITER = 150

print("Executing comprehensive IEEE scalability matrices...")
results_summary = {}

for size in TEST_SIZES:
    print(f"\n--- Processing QAP Instance Scale: N = {size} ---")
    flow, dist = generate_qap_matrices(size)
    num_bits = size * 8
    
    bpso_runs, qrao_runs = [], []
    for run in range(RUNS):
        bpso = VectorizedBinaryPSO(POP_SIZE, num_bits, MAX_ITER, size, flow, dist)
        qrao = FixedVectorizedQuantumRao(POP_SIZE, num_bits, MAX_ITER, size, flow, dist)
        bpso_runs.append(bpso.optimize())
        qrao_runs.append(qrao.optimize())
        
    try: _, p_val = wilcoxon(bpso_runs, qrao_runs)
    except ValueError: p_val = 1.0
        
    results_summary[size] = {
        'bpso_mean': np.mean(bpso_runs), 'qrao_mean': np.mean(qrao_runs),
        'bpso_std': np.std(bpso_runs), 'qrao_std': np.std(qrao_runs), 'p_value': p_val
    }

# Print the comprehensive multi-scale breakdown
print("\n" + "="*90)
print("                     FINAL IEEE SCALABILITY PROFILING TABLE")
print("="*90)
print(f"Scale (N) | BPSO Mean Cost       | Q-Rao Mean Cost      | Q-Rao Improvement | Wilcoxon p-value")
print("-"*90)
for size in TEST_SIZES:
    metrics = results_summary[size]
    imp = ((metrics['bpso_mean'] - metrics['qrao_mean']) / metrics['bpso_mean']) * 100
    print(f"N = {size:<5} | {metrics['bpso_mean']:<20.2f} | {metrics['qrao_mean']:<20.2f} | {imp:<17.2f}% | {metrics['p_value']:.4e}")
print("="*90)
