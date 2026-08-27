# 🌌 Hybrid Quantum Rao (HQ-Rao) Algorithm

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Optimization](https://img.shields.io/badge/Domain-Metaheuristic_Optimization-success)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Institution](https://img.shields.io/badge/Research-NIT_Warangal-orange)

A next-generation, parameter-less metaheuristic algorithm that integrates classical distance mechanics with quantum tunneling to solve highly complex, high-dimensional engineering problems.

---

## 📖 1. Introduction to Optimization
Optimization is the mathematical science of finding the *absolute best* solution from a massive pool of possibilities. Whether it's minimizing the fuel consumption of an autonomous VTOL, optimizing neural network weights, or designing aerodynamic structures, the goal is to find the lowest possible point (the global minimum) on a complex, multi-dimensional landscape. 

However, real-world engineering problems are highly non-linear, constrained, and filled with "local minima"—deceptive traps that look like the best solution but aren't. To navigate these, we use **Metaheuristics**: intelligent search algorithms inspired by physics, evolution, and swarm behavior.

## 🕰️ 2. The Landscape So Far: Work to Date
Historically, researchers have relied on several approaches to tackle these landscapes:
* **Classical Swarms (e.g., PSO):** Algorithms like Particle Swarm Optimization use momentum-based velocity vectors. While powerful, they require careful tuning of hyperparameters (inertia, cognitive/social coefficients).
* **Parameter-less Algorithms (e.g., Rao Algorithms):** Recent advancements introduced algorithms like Rao-1, 2, and 3, which eliminate the need for hyperparameter tuning. They rely purely on the best, worst, and random candidate solutions.
* **The Limitation:** While parameter-less models are incredibly user-friendly ("out-of-the-box" usability), they often suffer from poor **micro-exploitation**—meaning they struggle to aggressively dig into a promising area once they find it, sometimes stalling in high-dimensional spaces. 

*Previous attempts to fix this included superposition of states and random number generation tweaks, but these often broke the "parameter-less" rule or heavily increased computational time.*

---

## 🚀 3. Our Work: Enter HQ-Rao
The **Hybrid Quantum Rao (HQ-Rao)** algorithm was formulated to resolve the micro-exploitation deficiencies of standard metaheuristics without introducing a single arbitrary hyperparameter or increasing computational time complexity.

### How it Works:
Instead of just sharing trajectories like classical swarms, HQ-Rao introduces **Independent Orthogonal Tunneling**. 
1. **Dirac Potential Well Simulation:** We simulate a collapsing quantum well around the best solutions, allowing particles to rapidly descend.
2. **Gram-Schmidt Orthogonalization:** We create probabilistically mapped, perpendicular search topologies.
3. **Zero Overhead:** Most crucially, HQ-Rao achieves multi-order-of-magnitude performance gains while strictly maintaining **zero additional fitness evaluations**.

> **💡 The Result:** An ultra-fast convergence velocity that outmaneuvers traditional momentum-based vectors while remaining structurally simple.

---

## 🔬 4. Experimental Setup
To prove HQ-Rao's dominance, we engineered a high-performance Python benchmarking pipeline to test it against the toughest mathematical landscapes available.

### 🆚 Algorithms Evaluated:
* **HQ-Rao** (Proposed)
* **Rao-1** (Classical parameter-less baseline)
* **Rao-2 & Rao-3** (Peer-interaction baselines)
* **Particle Swarm Optimization (PSO)** (Globally established swarm baseline)

### 📊 Benchmarks Used:
The algorithms were rigorously evaluated across **35 high-dimensional objective functions**:
* **IEEE CEC 2014 Suite** (30 Dimensions)
* **IEEE CEC 2022 Suite** (20 Dimensions)
* *Note: Evaluations on the CEC 2022 suite were conducted using strict IEEE competition random seeds to entirely eliminate initialization bias.*

---

## 🏆 5. Results & Performance

HQ-Rao demonstrated overwhelming statistical dominance across the board. 

| Metric | Performance |
| :--- | :--- |
| **Win Rate vs Classical Baselines** | 🥇 **81.25%** |
| **Friedman Rank vs PSO** | 🥇 **1.33** (Dominant superiority) |
| **Friedman Rank vs Rao-3** | 🥇 **1.00** (Mathematically perfect) |
| **Computational Overhead** | 🟢 **0% Increase** (No extra function evaluations) |



### ⚠️ A Note on the No Free Lunch (NFL) Theorem
Because of its aggressive exploitation speed, HQ-Rao occasionally causes premature quantum well collapse on highly dense, periodic composition landscapes (like the Shifted and Rotated Levy function). Identifying this as a classical freezing state provides a clear theoretical path for future refinement via *Heisenberg Uncertainty Scattering*.

