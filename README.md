LINK FOR COMPARISON ALGORITHMS FOR MOBCO:
https://github.com/Samiksha-bajoria/Multi_objective_border_collie_optimization

# Multi-Objective Border Collie Optimization (MOBCO)

This repository contains the implementation and experimental framework for **Multi-Objective Border Collie Optimization (MOBCO)**, a multi-objective metaheuristic optimization algorithm inspired by the herding and hunting behavior of Border Collies.

The framework provides a complete pipeline for optimization, benchmark evaluation, performance analysis, visualization, sensitivity analysis, combined-fitness analysis, and statistical comparison with other multi-objective optimization algorithms.

---

## Overview

Multi-Objective Border Collie Optimization (MOBCO) extends the Border Collie Optimization (BCO) algorithm to solve multi-objective optimization problems.

The algorithm maintains a population of candidate solutions and uses Border Collie-inspired behaviors such as:

- Herding
- Stalking
- Gathering
- Eyeing
- Mutation
- Local search

MOBCO uses Pareto dominance and an external archive to maintain a diverse set of non-dominated solutions approximating the Pareto-optimal front.

---

## Features

The framework provides the following capabilities:

- Multi-objective Border Collie Optimization
- Pareto dominance-based solution selection
- Non-dominated sorting
- Crowding-distance based diversity preservation
- External archive management
- Herding and stalking mechanisms
- Gathering behavior
- Eyeing mechanism
- Adaptive velocity and acceleration
- Mutation and step-size decay
- Local search
- Hypervolume calculation
- IGD and GD calculation
- Spacing and Spread metrics
- Epsilon metric
- Runtime measurement
- Expected vs. obtained Pareto-front visualization
- Objective-space visualization
- Decision-space visualization
- Convergence analysis
- Sobol sensitivity analysis
- Combined-fitness analysis
- Statistical analysis using ANOVA
- Box-plot generation
- Support for multiple benchmark families
- Support for EvoPINN optimization problems

---

# Benchmark Problems

The framework supports **27 optimization problems** belonging to four major categories.

## EvoPINN Problems

The following 12 EvoPINN problems are supported:

1. EvoPINN1_Poisson1D
2. EvoPINN2_Heat1D
3. EvoPINN3_Advection1D
4. EvoPINN4_Wave1D
5. EvoPINN5_Burgers1D
6. EvoPINN6_AllenCahn1D
7. EvoPINN7_Poisson2D
8. EvoPINN8_ReactionDiffusion1D
9. EvoPINN9_DampedOscillator
10. EvoPINN10_VanDerPol
11. EvoPINN11_LotkaVolterra
12. EvoPINN12_Schrodinger1D

## ZDT Problems

Five standard ZDT benchmark problems are included:

1. ZDT1
2. ZDT2
3. ZDT3
4. ZDT4
5. ZDT6

## DTLZ Problems

Seven DTLZ benchmark problems are included:

1. DTLZ1
2. DTLZ2
3. DTLZ3
4. DTLZ4
5. DTLZ5
6. DTLZ6
7. DTLZ7

## Mixed Objective Problems

Three mixed-objective problems are included:

1. Mixed_ZDT1
2. Mixed_DTLZ1
3. Mixed_DTLZ2

---

# MOBCO Algorithm

The main optimization process consists of the following stages:

1. Initialize the population.
2. Evaluate objective functions.
3. Identify non-dominated solutions.
4. Update the external archive.
5. Calculate crowding distance.
6. Perform Border Collie-inspired herding behavior.
7. Update velocity, acceleration, and time.
8. Perform stalking and gathering operations.
9. Apply mutation and eyeing mechanisms.
10. Perform local search.
11. Re-evaluate the population.
12. Update the archive.
13. Repeat until the maximum number of iterations is reached.
14. Store the final Pareto-optimal solutions and performance metrics.

---

# Project Structure

```text
MOBCO_IEEE/
│
├── MOBCO/
│   ├── archive.py
│   ├── bounds_check.py
│   ├── check.py
│   ├── config.py
│   ├── crowding_distance.py
│   ├── dataset.py
│   ├── dominance.py
│   ├── evopinn_problem.py
│   ├── excel_writer.py
│   ├── fitness.py
│   ├── generate.py
│   ├── herding.py
│   ├── local_search.py
│   ├── main.py
│   ├── metrics.py
│   ├── mixed_test_functions.py
│   ├── nondominated_sort.py
│   ├── normalization.py
│   ├── optimizer.py
│   ├── population_init.py
│   ├── position_update.py
│   ├── rebuild_master_summary.py
│   ├── requirements.txt
│   ├── run_experiment.py
│   ├── sensitivity_analysis.py
│   ├── standard_test_functions.py
│   ├── test_mobco.py
│   ├── velocity_time_acceleration.py
│   └── visualization.py
│
├── Combination_graph/
│   ├── Combined_fitness.py
│   ├── Combined_plots_lib.py
│   └── requirements.txt
│
├── Anova_independent_runs_with_box_plots/
│   ├── requirements.txt
│   └── src/
│       ├── compute_combined_fitness.py
│       ├── export_combined_fitness_by_problem.py
│       ├── generate_boxplots.py
│       ├── load_data.py
│       └── run_anova_combined_fitness.py
│
├── LICENSE
└── README.md

# Performance Metrics

MOBCO is evaluated using multiple performance metrics to assess convergence, diversity, distribution, and computational efficiency.

## Hypervolume (HV)

Hypervolume measures the volume of the objective space dominated by the obtained Pareto front with respect to a reference point.

A higher Hypervolume value generally indicates better convergence and diversity.

## Inverted Generational Distance (IGD)

IGD measures the average distance between the obtained solution set and the reference Pareto-optimal front.

A lower IGD value indicates better convergence and coverage of the reference front.

## Generational Distance (GD)

GD measures the distance between the obtained Pareto front and the reference Pareto front.

A lower GD value indicates better convergence.

## Spacing

Spacing measures the uniformity of the distribution of solutions along the Pareto front.

A lower Spacing value generally indicates a more uniformly distributed solution set.

## Spread

Spread measures the extent and distribution of solutions across the Pareto front.

It provides information about how well the obtained solutions cover the objective space.

## Epsilon Indicator

The Epsilon indicator measures the minimum factor by which the obtained Pareto front must be translated to dominate the reference front.

A lower Epsilon value generally indicates better performance.

## Runtime

Runtime measures the computational time required by the algorithm to complete the optimization process.

---

# Visualization

The framework provides several visualization methods for analyzing the performance and behavior of MOBCO.

## Expected vs Obtained Pareto Front

The expected Pareto front is compared with the obtained non-dominated solutions to evaluate how closely MOBCO approaches the optimal solution set.

## Objective Space

Objective-space plots show the distribution of obtained solutions across the objective dimensions.

For two-objective problems, the solutions are represented using a 2D plot.

For three-objective problems, a 3D representation is used.

For higher-dimensional problems, suitable projections or parallel-coordinate representations are used.

## Decision Space

Decision-space visualization shows the distribution of candidate solutions in the original decision-variable space.

This helps analyze the diversity of solutions before they are mapped to the objective space.

## Convergence Plot

Convergence plots show the change in optimization performance over successive iterations.

The framework can visualize metrics such as:

- Hypervolume
- IGD
- GD

These plots help determine whether the algorithm is converging toward a stable solution set.

## Distribution Plot

Distribution plots are used to analyze the spread and uniformity of the obtained Pareto solutions.

---

# Sensitivity Analysis

A global sensitivity analysis is included to study the influence of MOBCO control parameters on optimization performance.

The analysis follows a variance-based Sobol sensitivity approach using Hypervolume (HV) as the primary response metric.

## Structural Parameters

The structural parameter group includes:

- Population Size
- Archive Size
- Mutation Rate
- Number of Dogs
- Eyeing-trigger Stagnation Length

## Dynamics-Shaping Parameters

The dynamics-shaping parameter group includes:

- Acceleration Damping
- Mutation Decay Floor
- Step Decay Floor
- Stalking Pull
- Gathering Pull

## Methodology

The sensitivity analysis uses Saltelli sampling to generate parameter combinations.

The resulting parameter configurations are evaluated using MOBCO, and the resulting Hypervolume values are used to calculate Sobol sensitivity indices.

The analysis helps identify:

- Parameters with the strongest influence on performance
- Parameters with relatively small effects
- First-order parameter effects
- Total-order parameter effects
- Parameter interaction effects

## Running Sensitivity Analysis
```bash
python sensitivity_analysis.py
