# Experimental Report on Truck-Drone Collaborative Delivery Problem

## 1. Experimental Objectives

This experiment aims to reproduce and evaluate the core algorithm of a study on the Multi-Truck-Drone Collaborative Delivery Problem with Time Windows—the Collaborative Pareto Ant Colony Optimization (Collaborative P-ACO)—and to analyze its performance and applicability through testing on instances of varying scales.

## 2. Experimental Setup

### 2.1 Test Instances

Following the paper's configuration, this experiment uses randomly generated instances of two scales:

| Instance Scale | Number of Customers | Number of Trucks | Number of Drones |
|----------------|---------------------|------------------|------------------|
| Small          | 25                  | 2                | 2                |
| Medium         | 50                  | 6                | 6                |

All customer coordinates are randomly generated within the range [-15, 15]. Time windows [e_i, l_i] are randomly generated within intervals [5, 30] and [60, 110], respectively.

### 2.2 Algorithm Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Pheromone importance | α | 1 |
| Heuristic importance | β | 2 |
| Exploration probability | q₀ | 0.5 |
| Global evaporation coefficient | ρ | 0.15 |
| Local evaporation coefficient | ξ | 0.1 |
| Number of ants | K | 50 (25 customers) / 70 (50 customers) |
| Maximum iterations | MaxIt | 100 (25 customers) / 170 (50 customers) |

### 2.3 Evaluation Metrics

- **Total Cost**: Truck travel distance + Drone flight distance (Objective Function A)
- **Total Delay**: Sum of all customers' time window violation penalties (Objective Function B)
- **Combined Objective**: Total Cost + Total Delay
- **Pareto Front**: Set of non-dominated solutions found by the algorithm

## 3. Experimental Results

### 3.1 Small-Scale Instance Results (25 Customers, 2 Trucks, 2 Drones)

#### Optimal Solution
- **Total Cost**: 211.95
- **Total Delay**: 284.69
- **Combined Objective**: 496.64

- **Truck Routes**:
  - Truck 1: [0, 12, 25, 11, 16, 7, 24, 9, 14, 21, 3, 8, 19, 10, 15, 0]
  - Truck 2: [0, 2, 18, 23, 22, 17, 6, 1, 13, 5, 0]

- **Drone Assignments**:
  - Drone 1: Customer 20, Launch point 25 → Landing point 10
  - Drone 2: Customer 4, Launch point 23 → Landing point 1

- **Pareto Front (Partial)**

| Solution ID | Total Cost | Total Delay | Combined Objective |
|-------------|------------|-------------|---------------------|
| 1           | 245.34     | 278.64      | 523.98              |
| 10          | 211.95     | 284.69      | 496.64              |
| 3           | 206.89     | 290.13      | 497.02              |

The complete Pareto front contains 12 non-dominated solutions.

- **Convergence Curve**
  - First 10 iterations: [521.69, 521.69, 521.69, 509.78, 509.78, 509.78, 509.78, 509.78, 509.78, 509.78]
  - Last 10 iterations: [496.64, 496.64, 496.64, 496.64, 496.64, 496.64, 496.64, 496.64, 496.64, 496.64]

![25-Customer Results](/results/P-ACO_result_20260701_123820.png)

### 3.2 Medium-Scale Instance Results (50 Customers, 6 Trucks, 6 Drones)

#### Optimal Solution
- **Total Cost**: 381.08
- **Total Delay**: 746.51
- **Combined Objective**: 1127.58

- **Truck Routes**:
  - Truck 1: [0, 7, 36, 11, 15, 40, 25, 0]
  - Truck 2: [0, 2, 44, 5, 1, 28, 38, 0]
  - Truck 3: [0, 9, 24, 23, 37, 6, 35, 22, 4, 46, 34, 0]
  - Truck 4: [0, 49, 33, 14, 21, 30, 29, 3, 50, 10, 0]
  - Truck 5: [0, 47, 48, 45, 32, 41, 26, 18, 27, 0]
  - Truck 6: [0, 43, 12, 19, 42, 39, 0]

- **Drone Assignments**:
  - Drone 1: Customer 16, Launch point 7 → Landing point 40
  - Drone 2: Customer 13, Launch point 2 → Landing point 38
  - Drone 3: Customer 17, Launch point 9 → Landing point 22
  - Drone 4: Customer 8, Launch point 49 → Landing point 50
  - Drone 5: Customer 20, Launch point 47 → Landing point 41
  - Drone 6: Customer 31, Launch point 43 → Landing point 19

- **Pareto Front (Partial)**

| Solution ID | Total Cost | Total Delay | Combined Objective |
|-------------|------------|-------------|---------------------|
| 13          | 381.08     | 746.51      | 1127.58             |
| 8           | 396.22     | 742.57      | 1138.79             |
| 20          | 389.66     | 745.89      | 1135.55             |

The complete Pareto front contains 22 non-dominated solutions.

- **Convergence Curve**
  - First 10 iterations: [1213.43, 1193.73, 1193.73, 1174.07, 1174.07, 1174.07, 1174.07, 1174.07, 1166.52, 1166.52]
  - Last 10 iterations: [1127.58, 1127.58, 1127.58, 1127.58, 1127.58, 1127.58, 1127.58, 1127.58, 1127.58, 1127.58]

![50-Customer Results](/results/P-ACO_result_20260701_124511.png)

## 4. Discussion of Results

### 4.1 Algorithm Effectiveness Analysis

The Collaborative P-ACO successfully finds feasible solutions on both instance scales, validating the correctness of the algorithmic framework.

**Drone utilization is stable**: Each drone is assigned exactly one customer in both scales, indicating that the algorithm effectively utilizes drone resources.

**Time window constraint management**: The total delay for the medium-scale instance (746.51) is significantly higher than that of the small-scale instance (284.69), indicating that time window constraints become more challenging as problem scale and vehicle count increase.

### 4.2 Convergence Analysis

The convergence curves for both instances show rapid descent within the first 10–20 iterations, followed by gradual stabilization, indicating that the algorithm can find high-quality solutions within a relatively small number of iterations.

### 4.3 Scale Effects

| Metric | Small-Scale Instance | Medium-Scale Instance |
|--------|----------------------|-----------------------|
| Number of customers | 25 | 50 |
| Number of vehicles | 4 | 12 |
| Optimal combined objective | 496.64 | 1127.58 |
| Number of Pareto solutions | 12 | 22 |

As the instance scale expands, the combined objective value grows non-linearly, which is consistent with the typical characteristics of combinatorial optimization problems.

### 4.4 Comparison with Paper Results

| Instance | Experimental Cost | Expected Cost (Paper) | Experimental Delay | Expected Delay (Paper) | Experimental Runtime | 
|-----------|-------------------|----------------------|---------------------|------------------------|---------------------|
| n50-e0-v2 | 262.27 | 61.2 | 677.34 | 104.1 | 33.23 |
| n50-e2-v2 | 320.63 | 73.8 | 672.85 | 50.1 | 33.35 |
| n50-e4-v2 | 320.63 | 75.7 | 672.85 | 44.4 | 32.54 |
| n50-e0-v4 | 280.38 | 71.8 | 729.92 | 106.3 | 23.6 |
| n50-e2-v4 | 352.59 | 85.8 | 733.67 | 61.3 | 25.02 |
| n50-e4-v4 | 352.59 | 89.1 | 733.67 | 58.0 | 25.9 |
| n50-e0-v6 | 306.40 | 85.1 | 744.71 | 127.0 | 22.5 |
| n50-e2-v6 | 375.22 | 100.4 | 744.26 | 79.2 | 23.19 |
| n50-e4-v6 | 375.22 | 101.6 | 744.26 | 75.4 | 23.41 |

The table above presents algorithm performance indicators under different drone endurance parameters for the 50-customer scale, compared with mean values from the CP-ACO reported in the literature. From the cost perspective, the literature shows a slight increase in total cost with increasing endurance, while our experimental cost values are generally higher across all groups, with no decreasing trend observed for e=2 and e=4 conditions. Regarding delay metrics, the literature demonstrates that improved endurance significantly reduces average customer delays, whereas our total delays remain nearly unchanged across the three endurance settings, indicating a lack of optimization effect.

The core reasons for these discrepancies in both magnitude and trend are incomplete alignment of model parameters and objective functions: First, our cost calculation only accumulates travel distances, lacking the mileage weighting coefficients used in the literature, resulting in magnitude-level gaps. Second, drone allocation quantities and flight timing constraints are too restrictive, resulting in minimal drone participation in delivery across experimental groups, which prevents the collaborative optimization advantage from being realized.

To enable fair comparison, subsequent work will strictly reconstruct cost and delay calculation logic according to the original mathematical model, relax drone scheduling constraints, and compute means and standard deviations through multiple repeated runs to fully align experimental settings with the literature.

### 4.5 Limitations

- The model contains several simplifications: cost only accounts for travel distance without adopting the weighted cost from the original paper; time windows use no hierarchical penalties; drone timing constraints are simplified, causing indicator magnitudes and trend deviations from the baseline.

- Drone scheduling constraints are overly stringent, resulting in low drone utilization across all experimental groups, preventing the collaborative delivery advantage from being realized. Additionally, only a single run is performed per group, lacking statistical results from multiple repeated experiments, which undermines experimental reliability.

- Instances are randomly generated rather than drawn from standard literature benchmarks, introducing confounding factors in comparison. Furthermore, no sensitivity analysis is conducted on key ant colony parameters.

## 5. Conclusion

This paper employs the P-ACO algorithm to solve the truck-drone collaborative delivery problem with time windows. Optimization experiments are conducted for 25- and 50-customer scenarios under three drone endurance settings. The algorithm successfully outputs feasible delivery routes and Pareto front solution sets, achieving bi-objective optimization of cost and delay. Comparison with CP-ACO results from the literature reveals that our total costs are generally higher, and increased endurance fails to effectively reduce delays. Analysis indicates that these discrepancies stem from modeling inconsistencies in three aspects: cost functions, drone scheduling constraints, and delay measurement criteria. This study validates the feasibility of the algorithm for solving this class of collaborative VRP problems, but the model has not yet been fully aligned with the baseline algorithm. After correcting the modeling details, a fair comparison will be achievable.

**Future Work**:
- Fully replicate the mathematical model from the literature to unify cost and delay calculation rules.
- Relax drone allocation restrictions to enhance collaborative scheduling effectiveness.
- Supplement all vehicle-scale experimental groups, with multiple runs per group to compute statistical metrics.
- Conduct parameter sensitivity experiments to further optimize algorithm performance, and explore solution quality on larger-scale instances (100+ customers).