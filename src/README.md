# `README.md` 

# Hybrid P-ACO-PLS for TDRPTW
Multi-objective hybrid algorithm combining Pareto Ant Colony Optimization and Pareto Local Search to solve Truck-Drone Routing Problem with Time Windows.

## Overview
This project proposes a two-stage framework:
1. Parallel multi-mode P-ACO for global solution construction
2. Periodic PLS with five neighborhood operators for local refinement

Experiments on 45 test instances prove the hybrid method improves solution quality up to 10.1% and boosts Pareto diversity by 459%, with 100% feasibility rate.

## Repository Structure
```
src/
├── final/                     # Core codes, results & full report
│   ├── compare_results_e2/    # Plots and results for endurance=2
│   ├── compare_results_e4/    # Plots and results for endurance=4
│   ├── compare_results_e6/    # Plots and results for endurance=6
│   ├── images/                # General project figures, charts & diagrams
│   ├── results/               # Raw experimental outputs
│   ├── config.py              # Global hyperparameters
│   ├── final_report.md        # Full formal report
│   ├── instance_generator.py  # Instance generation script
│   ├── original_P-ACO.py      # Baseline algorithm
│   ├── P-ACO_PLS.py           # Proposed hybrid algorithm
│   └── run_all.py             # Batch run all tests
└── week01 ~ week06/           # Weekly development records
```

## Environment & Dependencies
- Python 3.11.15
- NumPy 2.4.6, Matplotlib 3.11.0

## Full Report
Complete theory, mathematical formulation, experimental results and discussion:
[final/final_report.md](final/final_report.md)

## Key Results
- Objective improvement: 3.9% ~ 10.1% under different drone endurance limits
- Pareto front size increased by up to 459%
- 100% feasible solutions across all test cases
- Up to 45.8% runtime reduction on large-scale instances

## References

1. D. N. Das, R. Sewani, J. Wang and M. K. Tiwari, "Synchronized Truck and Drone Routing in Package Delivery Logistics," in IEEE Transactions on Intelligent Transportation Systems, vol. 22, no. 9, pp. 5772-5782, Sept. 2021.
2. Q. Luo, G. Wu, B. Ji, L. Wang and P. N. Suganthan, "Hybrid Multi-Objective Optimization Approach With Pareto Local Search for Collaborative Truck-Drone Routing Problems Considering Flexible Time Windows," in IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 8, pp. 13011-13025, Aug. 2022.