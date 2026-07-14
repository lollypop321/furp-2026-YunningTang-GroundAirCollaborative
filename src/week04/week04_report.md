# Algorithm Improvement Report: Delay-Prioritized Drone Assignment Strategy with Metropolis Criterion Integration

---

## 1. Overview of Improvements

This report documents two-stage enhancements made to the P-ACO (Pareto-based Ant Colony Optimization) algorithm for truck-drone collaborative delivery systems:

1. **Delay-Prioritized Drone Assignment Strategy**: Shifts the optimization focus from cost savings to time-window compliance by redesigning customer prioritization rules and scoring functions to serve time-urgent customers first.

2. **Metropolis Criterion Integration**: Incorporates the Metropolis criterion from simulated annealing into the pheromone update phase, probabilistically accepting suboptimal solutions to enhance global search capability and prevent premature convergence to local optima.

---

## 2. Delay-Prioritized Drone Assignment Strategy

### 2.1 Customer Prioritization Enhancement

**Original Logic**: Customers were sorted by time window width (`l - e`), which only reflects the looseness of time windows without capturing actual urgency.

**Improved Logic**: A dynamic urgency scoring function is introduced to prioritize customers based on the remaining time until their service deadlines. Specifically, the algorithm calculates each customer's estimated arrival time if served by a truck, then computes the remaining time until the deadline. Customers with less remaining time receive higher urgency scores and are prioritized for drone assignment. Customers already overdue are assigned the maximum urgency score.

### 2.2 Scoring Function Redesign

**Original Scoring Function**: Combined cost savings and time improvement with a weight ratio of approximately 1:0.5, favoring cost reduction.

**Improved Scoring Function**:

- **Tardiness Improvement Calculation**: The algorithm separately computes the expected tardiness if the customer is served by a truck versus by a drone. The reduction achieved through drone assignment is the tardiness improvement.
  
- **Weight Adjustment**: The weight for tardiness improvement is increased from 0.5 to 10.0, while the weight for cost savings remains 1.0, resulting in a 10:1 ratio favoring delay reduction.

- **Bonus Mechanism**: An additional reward is granted when drone service completely eliminates tardiness that would have occurred with truck service.

### 2.3 Effect of the Improvement

This strategy enables the algorithm to fully consider time-window constraints during the path construction phase, prioritizing drone resources for time-urgent customers. This significantly reduces total tardiness while maintaining competitive cost performance.

---

## 3. Metropolis Criterion Integration

### 3.1 Core Mechanism

Building upon the delay-prioritized framework, the Metropolis criterion is additionally applied during the pheromone update phase. After obtaining all feasible solutions in each iteration, in addition to unconditionally updating the pheromone of the optimal solution, each suboptimal solution is evaluated based on its delay difference from the optimal solution. The probability of accepting a suboptimal solution's pheromone contribution is `exp(-Δdelay / T)`, where T is the current temperature parameter.

This mechanism allows solutions with slightly inferior delay performance but potentially valuable structural information to still influence subsequent searches, thereby maintaining population diversity.

### 3.2 Parameter Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Initial Temperature | 100.0 | High acceptance probability at algorithm start |
| Minimum Temperature | 0.1 | Prevents temperature from reaching zero |
| Cooling Rate | 0.98 | Exponential decay per iteration |
| Suboptimal Update Intensity | 0.5× of optimal | Controls influence of suboptimal solutions on pheromone |

### 3.3 Pheromone Update Process

1. Optimal solution pheromone is updated unconditionally with 1.0× intensity
2. For each feasible solution (excluding the optimal):
   - Calculate delay difference Δdelay
   - If Δdelay ≤ 0, treat as superior solution and update normally
   - If Δdelay > 0, accept with probability exp(-Δdelay/T) and update with 0.5× intensity
3. Temperature is reduced after each iteration: `T = max(min_temp, T × cooling_rate)`

### 3.4 Effect of the Improvement

The integration of the Metropolis criterion allows the algorithm to maintain exploration capability even in later search stages, preventing entrapment in local optima due to over-exploitation of the current best solution. This enhances algorithmic robustness across a wider range of parameter configurations.

---

## 4. Experimental Results

### 4.1 Overall Performance Comparison (Varying ENDURANCE)

| ENDURANCE | Algorithm Version | Total Cost | Total Delay | Combined Objective | Feasible Solutions | Runtime (s) |
|-----------|-------------------|------------|-------------|-------------------|-------------------|-------------|
| 0 | Delay-Prioritized | 163.95 | 137.81 | 301.75 | ✓ | 24.41 |
| 0 | Delay-Prioritized + Metropolis | 164.44 | 141.19 | 305.62 | ✓ | 23.22 |
| 2 | Delay-Prioritized | 171.05 | 127.12 | 298.17 | ✓ | 24.77 |
| 2 | Delay-Prioritized + Metropolis | 160.09 | 132.94 | 293.03 | ✓ | 23.02 |
| **4** | **Delay-Prioritized** | **175.74** | **115.27** | **291.01** | **✓** | **23.58** |
| **4** | **Delay-Prioritized + Metropolis** | **171.60** | **107.76** | **279.36** | **✓** | **21.85** |
| 6 | Delay-Prioritized | 194.29 | 111.36 | 305.65 | ✓ | 23.28 |
| 6 | Delay-Prioritized + Metropolis | 203.95 | 108.30 | 312.25 | ✓ | 21.72 |
| 8 | Delay-Prioritized | 192.44 | 97.00 | 289.44 | ✓ | 23.53 |
| 8 | Delay-Prioritized + Metropolis | 194.23 | 98.13 | 292.36 | ✓ | 22.21 |

![Metropolis Results Example (ENDURANCE = 6)](/src/week04/results/P-ACO_result_20260714_153451.png)

### 4.2 Optimal Performance Summary

| Configuration | Best Version | Combined Objective |
|---------------|--------------|-------------------|
| ENDURANCE = 0 | Delay-Prioritized | 301.75 |
| ENDURANCE = 2 | Delay-Prioritized + Metropolis | 293.03 |
| ENDURANCE = 4 | Delay-Prioritized + Metropolis | 279.36 |
| ENDURANCE = 6 | Delay-Prioritized | 305.65 |
| ENDURANCE = 8 | Delay-Prioritized | 289.44 |

---

## 5. Current Problems and Challenges

### 5.1 Inconsistent Performance of Metropolis Integration

While the Metropolis-enhanced version demonstrates significant improvements at ENDURANCE=2 and 4, it exhibits performance degradation at ENDURANCE=0, 6, and 8:

| ENDURANCE | Delay-Prioritized | Metropolis-Enhanced | Difference |
|-----------|-------------------|--------------------|------------|
| 0 | 301.75 | 305.62 | **+1.3% (worse)** |
| 2 | 298.17 | 293.03 | **-1.7% (better)** |
| 4 | 291.01 | 279.36 | **-4.0% (better)** |
| 6 | 305.65 | 312.25 | **+2.2% (worse)** |
| 8 | 289.44 | 292.36 | **+1.0% (worse)** |

This inconsistency indicates that the Metropolis criterion does not consistently improve solution quality across all parameter configurations. The probabilistic acceptance of suboptimal solutions, while beneficial for maintaining diversity, can sometimes introduce excessive noise that distracts the search from promising solution regions.

### 5.2 Temperature Scheduling Sensitivity

The current cooling schedule (initial temperature = 100.0, cooling rate = 0.98) may not be optimal for all ENDURANCE configurations. When drone endurance is low (0), the search space is more constrained, and aggressive exploration may be counterproductive. Conversely, when endurance is high (6, 8), the solution space expands, and the current cooling rate may cool too quickly, limiting the benefits of the Metropolis mechanism.

### 5.3 Suboptimal Solution Acceptance Threshold

The current implementation accepts any suboptimal solution with probability `exp(-Δdelay/T)`, regardless of the magnitude of Δdelay. This may lead to the acceptance of solutions with significantly worse delay performance, particularly at high temperatures. A threshold-based acceptance mechanism could potentially improve performance by filtering out solutions that deviate too far from the optimal.

### 5.4 Pheromone Update Intensity

Suboptimal solutions currently update pheromone at 0.5× intensity. This fixed ratio may not be optimal across different configurations. A dynamic intensity that adapts based on solution quality or search progress could yield better results.

---

## 6. Future Work and Improvements

### 6.1 Adaptive Temperature Scheduling

Replace the fixed exponential cooling schedule with an adaptive mechanism that adjusts the cooling rate based on search progress:

- **Performance-based Cooling**: If the algorithm has not improved for several iterations, slow down cooling to encourage more exploration
- **Configuration-aware Initial Temperature**: Set initial temperature proportional to the expected solution variance, which may vary with ENDURANCE

### 6.2 Threshold-based Acceptance

Introduce a threshold mechanism that only accepts suboptimal solutions when their delay degradation is within a reasonable range:

```
if Δdelay < α × best_delay:
    accept with probability exp(-Δdelay/T)
else:
    reject
```

Where α is a configurable parameter (e.g., 0.2) that defines the maximum acceptable degradation relative to the current best solution.

### 6.3 Dynamic Pheromone Update Intensity

Replace the fixed 0.5× intensity with a dynamic value that depends on:

- **Relative Solution Quality**: Solutions closer to the optimal receive higher update intensity
- **Search Progress**: Higher exploration in early iterations, higher exploitation in later iterations
- **Temperature**: Intensity could be inversely proportional to temperature

### 6.4 Metropolis Application Strategy

Consider alternative ways to apply the Metropolis criterion:

- **Solution-level Application**: Apply Metropolis during solution construction, not just during pheromone update
- **Hybrid Strategy**: Use Metropolis only when the algorithm shows signs of stagnation (e.g., no improvement for N iterations)

### 6.5 Parameter Optimization

Systematically optimize Metropolis parameters (initial temperature, cooling rate, acceptance threshold, update intensity) for different ENDURANCE configurations using design of experiments or Bayesian optimization techniques.

### 6.6 Multi-Objective Consideration

Extend the Metropolis criterion to consider both cost and delay simultaneously. The current approach considers only delay difference, but a multi-dimensional acceptance criterion could provide better guidance:

```
acceptance_prob = exp(-(α×Δcost + β×Δdelay) / T)
```

---

## 7. Conclusion

Through two stages of progressive improvements, this study draws the following conclusions:

1. **Effectiveness of Metropolis Integration**: At ENDURANCE=4, the Metropolis-enhanced version achieves a 4.0% improvement in the combined objective compared to the delay-prioritized version, with reductions of 6.5% in delay and 2.4% in cost. Runtime is also reduced by 7.3%, demonstrating both solution quality and efficiency gains.

2. **Conditional Advantages**: The Metropolis-enhanced version excels at moderate endurance configurations (2 and 4), where the search space is most complex and diversity maintenance provides the greatest benefit. The delay-prioritized strategy performs better at extreme configurations.

3. **Current Limitations**: The Metropolis integration does not consistently improve performance across all configurations. At ENDURANCE=0, 6, and 8, the combined objective is worse than the delay-prioritized version. This suggests that the current Metropolis parameters and implementation require further refinement.

4. **Future Directions**: Adaptive temperature scheduling, threshold-based acceptance, dynamic pheromone update intensity, and systematic parameter optimization are promising directions for improving the robustness and effectiveness of the Metropolis-enhanced algorithm.

5. **Reference Value**: Inspired by the MACO algorithm of Li et al. (2020), this work successfully transfers the Metropolis criterion from the UAV path planning domain to the truck-drone collaborative delivery scenario, extending the applicability of the method.

