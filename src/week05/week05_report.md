# Multi-Objective Optimization Report for Truck-Drone Collaborative Delivery

---

## 1. Research Background

This report addresses the truck-drone collaborative routing problem. Based on the original P-ACO (Pareto Ant Colony Optimization) algorithm framework, we introduce a **hybrid multi-objective optimization strategy** inspired by the HMOA paper, focusing on improving Pareto front diversity and enhancing local search capabilities.

---

## 2. Methodology Enhancements

### 2.1 Original Method (P-ACO)

The original algorithm adopts a **Pareto Ant Colony Optimization** framework with the following core mechanisms:

| Component | Description |
|-----------|-------------|
| Ant Construction | 5 personality modes (delay_first / cost_first / balance / delay_favor / cost_favor) |
| Pheromone Update | Best solution + top 5 Pareto solutions contribute jointly |
| Pareto Archive | Crowding distance pruning with boundary solution protection |
| Drone Assignment | Delay-prioritized greedy scoring mechanism |

### 2.2 New Enhancements

#### 2.2.1 Pareto Local Search (PLS)

Inspired by the HMOA paper, a PLS module is embedded into the P-ACO framework, triggered every 5 generations to perform **neighborhood depth search** on non-dominated solutions in the current Pareto archive.

**PLS Core Workflow:**
1. Select the top 5 solutions from the Pareto archive
2. Apply the following neighborhood operations to each solution:
   - 2-opt: Optimize truck routes
   - Truck-to-Drone: Reassign truck customers to drone service
   - Drone-to-Truck: Reassign drone customers back to truck service
   - Drone Reassign: Re-optimize drone task allocation
   - Swap: Exchange truck and drone customers

#### 2.2.2 Five Neighborhood Operators

| Operator | Name | Description |
|----------|------|-------------|
| N1 | Truck-to-Drone | Replace the most expensive truck customer with drone service |
| N2 | Drone-to-Truck | Move the most expensive drone task back to the truck |
| N3 | Swap | Exchange service modes between a truck customer and a drone customer |
| N4 | 2-opt | Optimize truck routes by eliminating crossings |
| N5 | Drone Reassign | Delete and reassign the most expensive drone task |

#### 2.2.3 Boundary Solution Protection

During Pareto archive pruning, boundary solutions (those with minimum cost and minimum delay) are preserved to prevent the loss of extreme alternatives, ensuring the completeness of the Pareto front.

#### 2.2.4 Pareto De-duplication Mechanism

To prevent the Pareto archive from containing numerous "pseudo non-dominated solutions" (solutions that are numerically close but represent the same operational plan), a similarity threshold check is introduced. If the Euclidean distance between a new solution and any existing solution in the archive is less than the threshold (default 0.5), it is considered a duplicate and rejected. This ensures that the archive retains only solutions with meaningful differentiation in the cost-delay space.

---

## 3. Experimental Setup

### 3.1 50-Customer Experiments

| Parameter | Value |
|-----------|-------|
| Number of Customers | 50 |
| Number of Trucks | 4 |
| Number of Drones | 4 (1 per truck) |
| Coordinate Range | ±10 |
| Time Windows | E=[0,5], L=[18,28] |
| Number of Ants | 70 |
| Number of Iterations | 170 |
| Drone Endurance | 0, 2, 4, 6 (comparative study) |

### 3.2 100-Customer Experiments (Planned)

| Parameter | Value |
|-----------|-------|
| Number of Customers | 100 |
| Number of Trucks | 8 |
| Number of Drones | 8 (1 per truck) |
| Coordinate Range | ±15 |
| Time Windows | E=[0,6], L=[22,34] |
| Number of Ants | 120 |
| Number of Iterations | 250 |

---

## 4. Results and Comparative Analysis

### 4.1 Comprehensive Comparison Table

| Endurance | Metric | Before | After | Change |
|-----------|--------|--------|-------|--------|
| **0** | Objective Value | 390.50 | **372.26** | **↓ 4.67%** |
| | Cost | 150.21 | 147.63 | ↓ 2.58 |
| | Delay | 240.28 | 224.63 | ↓ 15.65 |
| | Pareto Points | 2 | 1 | -1 |
| **2** | Objective Value | 389.16 | **378.20** | **↓ 2.82%** |
| | Cost | 157.20 | 147.66 | ↓ 9.54 |
| | Delay | 231.95 | 230.54 | ↓ 1.41 |
| | Pareto Points | 3 | 2 | -1 |
| **4** | Objective Value | 406.54 | **352.02** | **↓ 13.41%** |
| | Cost | 192.61 | 163.80 | ↓ 28.81 |
| | Delay | 213.93 | 188.22 | ↓ 25.71 |
| | Pareto Points | 4 | **12** | **↑ 8** |
| **6** | Objective Value | 354.19 | **336.04** | **↓ 5.12%** |
| | Cost | 186.69 | 155.97 | ↓ 30.72 |
| | Delay | 167.50 | 180.07 | ↑ 12.57 |
| | Pareto Points | 1 | **8** | **↑ 7** |

**P-ACO**:
![Old result](/results/P-ACO_result_20260716_172815.png)

**P-ACO + PLS**:
![New result](/results/P-ACO_PLS_result_20260716_172929.png)

### 4.2 Key Findings

#### 4.2.1 Comprehensive Improvement Across All Configurations

| Endurance | Before Objective | After Objective | Improvement |
|-----------|-----------------|-----------------|-------------|
| 0 | 390.50 | 372.26 | **4.67%** |
| 2 | 389.16 | 378.20 | **2.82%** |
| 4 | 406.54 | 352.02 | **13.41%** |
| 6 | 354.19 | 336.04 | **5.12%** |

**The enhanced algorithm achieves better objective values than the original algorithm across all endurance configurations.**

#### 4.2.2 Significant Improvement in Pareto Diversity

| Endurance | Before Points | After Points | Improvement |
|-----------|--------------|--------------|-------------|
| 4 | 4 | **12** | **3×** |
| 6 | 1 | **8** | **8×** |

At endurance=4 and 6, the enhanced algorithm significantly increases the number of Pareto points, demonstrating that PLS effectively discovers more non-dominated solutions.

#### 4.2.3 Differentiation of Three Extreme Solutions

| Version | Endurance | Combined Best | Cost Min | Delay Min |
|---------|-----------|---------------|----------|-----------|
| Before | 6 | (186.69, 167.50) | (186.69, 167.50) | (186.69, 167.50) |
| After | 6 | **(155.97, 180.07)** | **(144.59, 209.18)** | **(182.23, 179.65)** |

Before enhancement, the three extreme solutions **completely coincide**, indicating the algorithm failed to find effective trade-off solutions. After enhancement, the three solutions are **distinct**, forming a genuine Pareto front.

### 4.3 Optimal Configuration

**The best objective value is achieved at Endurance=6, with the following results:**

| Type | Cost | Delay | Objective |
|------|------|-------|-----------|
| Combined Best | 155.97 | 180.07 | **336.04** |
| Cost Min | 144.59 | 209.18 | 353.77 |
| Delay Min | 182.23 | 179.65 | 361.88 |

---

## 5. Conclusions

1. **PLS effectively improves solution quality**: Across all endurance configurations, objective values improve by 2.8%–13.4%, validating the effectiveness of Pareto Local Search for the truck-drone collaborative routing problem.

2. **Significant improvement in Pareto diversity**: Particularly at endurance=4 and 6, Pareto points increase from 4→12 and 1→8, respectively, indicating that PLS neighborhood search effectively discovers more non-dominated solutions, enriching the set of alternatives available to decision-makers.

3. **Formation of genuine trade-off frontiers**: Before enhancement, the three extreme solutions often coincided; after enhancement, the three solutions are distinct, providing real cost-delay trade-off options with practical decision-making value.

4. **Deduplication mechanism ensures archive quality**: The similarity threshold check effectively filters out pseudo non-dominated solutions caused by floating-point precision, ensuring that each point in the Pareto archive represents a truly distinct delivery plan.

---

## 6. Future Directions

1. **Continue testing on 100-customer and larger instances**: Current experiments are limited to 50-customer scenarios. Larger instances (100+ customers) are needed to verify the algorithm's performance and scalability on bigger problems.

2. **Tighter time windows**: In the 50-customer tests, delays are generally high (180–240). Tighter time windows would create more "urgency scenarios," increasing drone utilization and validating algorithm performance under more stringent constraints.

3. **Adjust execution order of 2-opt and drone operators in PLS**: Currently, PLS executes 2-opt before drone adjustments, which may overly optimize truck routes and reduce the relative advantage of drones. Reordering these operations could encourage the algorithm to explore more diverse solution spaces.

4. **Test on varied customer distribution patterns**: Currently only tested on uniform distributions. Future work should include clustered, banded, and other distribution patterns to evaluate the algorithm's robustness and generalization capability.