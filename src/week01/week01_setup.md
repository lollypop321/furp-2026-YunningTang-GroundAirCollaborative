# Week 1 Lab: Solver Setup and Baseline Smoke Test

## Objectives

By the end of Week 1, each team should have one small routing instance solved by a baseline tool, with objective value and feasibility status recorded.

## Step 1: Choose a Starter Path

Pick one:

1. **OR-Tools VRPTW path** for learning constraints and solver output.


## Step 2: Environment Record

Record:

- operating system: macOS 15.0.1;
- Python version: 3.11.15;
- package manager: pip 25.1;
- solver or codebase version: OR-Tools 9.15.6755;
- exact install commands: pip install ortools;
- hardware used for runtime: Apple M1 8GB.

## Step 3: Smoke Test

[My smoke test code](smoke_test.py)

Run one tiny instance and save:

- command;
- instance name and size;
- objective value;
- feasibility status;
- runtime;
- route plot or textual route output.

The route does not need to be optimal. It must be executable and interpretable.

## Step 4: Reflection

Write one short paragraph:

- What constraint was easiest to understand?
- What output was confusing?
- What will be the Week 2 baseline target?

The most straightforward constraint to understand was the basic vehicle routing cost minimization defined by the distance matrix, as it directly maps to the core goal of finding the shortest possible route. For Week 2, my baseline target is to adapt this smoke test script to a slightly larger instance with 5–10 nodes, add basic vehicle capacity constraints, and ensure the solver can still produce a feasible solution with clear, reproducible output.

## Deliverables

- Completed [Week 1 checkpoint](week01_checkpoint.md).
- Baseline command and output.
- Objective value, feasibility status, and runtime.
- One screenshot, plot, or route text.
