# Multi-Objective Border Collie Optimization (MOBCO)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Research](https://img.shields.io/badge/IEEE%20CIS-Research%20Internship-orange.svg)](#)

> Research Internship work carried out under **IEEE CIS** on the design and implementation of **Multi-Objective Border Collie Optimization (MOBCO)**.

This repository contains the full implementation and experimental framework for **MOBCO**, a multi-objective metaheuristic optimization algorithm inspired by the herding and hunting behavior of Border Collies. It provides an end-to-end pipeline covering optimization, benchmark evaluation, performance analysis, visualization, sensitivity analysis, combined-fitness analysis, and statistical comparison against other multi-objective optimization algorithms.

A companion repository with comparison algorithms used to benchmark MOBCO is available here: **[Multi-Objective Border Collie Optimization — Comparison Algorithms](https://github.com/Samiksha-bajoria/Multi_objective_border_collie_optimization)**.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [The MOBCO Algorithm](#the-mobco-algorithm)
- [Benchmark Problems](#benchmark-problems)
- [Performance Metrics](#performance-metrics)
- [Visualization](#visualization)
- [Sensitivity Analysis](#sensitivity-analysis)
- [Statistical Analysis](#statistical-analysis)
- [Related Work](#related-work)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

**Multi-Objective Border Collie Optimization (MOBCO)** extends the single-objective Border Collie Optimization (BCO) algorithm to solve multi-objective optimization problems.

The algorithm maintains a population of candidate solutions and models Border Collie-inspired behaviors such as:

- **Herding** – guiding the population toward promising regions of the search space
- **Stalking** – controlled, cautious movement toward target solutions
- **Gathering** – convergence of the population around non-dominated solutions
- **Eyeing** – a stagnation-triggered mechanism to maintain search pressure
- **Mutation** – stochastic perturbation to preserve diversity
- **Local search** – fine-grained refinement of candidate solutions

MOBCO uses **Pareto dominance** together with an **external archive** to maintain a diverse, well-distributed set of non-dominated solutions approximating the true Pareto-optimal front.

---

## Features

- Multi-objective Border Collie Optimization core algorithm
- Pareto dominance-based solution selection
- Non-dominated sorting
- Crowding-distance-based diversity preservation
- External archive management
- Herding, stalking, and gathering mechanisms
- Eyeing (stagnation-triggered) mechanism
- Adaptive velocity and acceleration updates
- Mutation and step-size decay
- Local search refinement
- Hypervolume (HV), IGD, and GD metric calculation
- Spacing, Spread, and Epsilon indicator calculation
- Runtime measurement
- Expected vs. obtained Pareto-front visualization
- Objective-space and decision-space visualization
- Convergence and distribution analysis
- Sobol sensitivity analysis (Saltelli sampling)
- Combined-fitness analysis
- Statistical analysis via ANOVA with box-plot generation
- Support for 27 benchmark problems across four problem families, including EvoPINN-based problems

---

## Repository Structure

```
MOBCO_IEEE/
│
├── MOBCO_git/                              # Core MOBCO algorithm implementation
│   ├── archive.py                          # External archive management
│   ├── bounds_check.py                     # Decision-variable bound enforcement
│   ├── check.py                            # Validation utilities
│   ├── config.py                           # Algorithm and experiment configuration
│   ├── crowding_distance.py                # Crowding-distance calculation
│   ├── dataset.py                          # Dataset / problem loading utilities
│   ├── dominance.py                        # Pareto dominance checks
│   ├── evopinn_problem.py                  # EvoPINN benchmark problem definitions
│   ├── excel_writer.py                     # Results export to Excel
│   ├── fitness.py                          # Objective function evaluation
│   ├── generate.py                         # Population / solution generation
│   ├── herding.py                          # Herding behavior
│   ├── local_search.py                     # Local search refinement
│   ├── main.py                             # Main entry point
│   ├── metrics.py                          # Performance metric calculations
│   ├── mixed_test_functions.py             # Mixed-objective benchmark problems
│   ├── nondominated_sort.py                # Non-dominated sorting
│   ├── normalization.py                    # Objective-space normalization
│   ├── optimizer.py                        # Core optimization loop
│   ├── population_init.py                  # Population initialization
│   ├── position_update.py                  # Position/velocity update rules
│   ├── rebuild_master_summary.py           # Summary report regeneration
│   ├── requirements.txt                    # Dependencies for the core algorithm
│   ├── run_experiment.py                   # Experiment runner
│   ├── sensitivity_analysis.py             # Sobol sensitivity analysis
│   ├── standard_test_functions.py          # ZDT / DTLZ benchmark problems
│   ├── test_mobco.py                       # Test suite
│   ├── velocity_time_acceleration.py       # Velocity/acceleration/time dynamics
│   └── visualization.py                    # Plotting and visualization utilities
│
├── Combination_graph/                      # Combined-fitness analysis
│   ├── Combined_fitness.py
│   ├── Combined_plots_lib.py
│   └── requirements.txt
│
├── Anova_independent_runs_with_box_plots/  # Statistical comparison across runs
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
```

---

## Getting Started

### Prerequisites

- Python 3.x
- Dependencies listed in each module's `requirements.txt`

### Installation

```bash
git clone https://github.com/Supriti2206/MOBCO_Multi_Objective_Border_Collie_Optimization_IEEE.git
cd MOBCO_Multi_Objective_Border_Collie_Optimization_IEEE

# Install dependencies for the core algorithm
pip install -r MOBCO_git/requirements.txt

# Install dependencies for combined-fitness analysis (optional)
pip install -r Combination_graph/requirements.txt

# Install dependencies for ANOVA / box-plot analysis (optional)
pip install -r Anova_independent_runs_with_box_plots/requirements.txt
```

---

## Usage

### Running the Optimizer

```bash
cd MOBCO_git
python main.py
```

### Running a Full Experiment

```bash
python run_experiment.py
```

### Running Sensitivity Analysis

```bash
python sensitivity_analysis.py
```

### Running Combined-Fitness Analysis

```bash
cd Combination_graph
python Combined_fitness.py
```

### Running Statistical (ANOVA) Analysis

```bash
cd Anova_independent_runs_with_box_plots/src
python run_anova_combined_fitness.py
```

> Configuration options such as population size, number of iterations, archive size, and problem selection can be adjusted in `MOBCO_git/config.py`.

---

## The MOBCO Algorithm

The main optimization process follows these stages:

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

## Benchmark Problems

The framework supports **27 optimization problems** across four major categories.

### EvoPINN Problems (12)

| # | Problem |
|---|---------|
| 1 | EvoPINN1_Poisson1D |
| 2 | EvoPINN2_Heat1D |
| 3 | EvoPINN3_Advection1D |
| 4 | EvoPINN4_Wave1D |
| 5 | EvoPINN5_Burgers1D |
| 6 | EvoPINN6_AllenCahn1D |
| 7 | EvoPINN7_Poisson2D |
| 8 | EvoPINN8_ReactionDiffusion1D |
| 9 | EvoPINN9_DampedOscillator |
| 10 | EvoPINN10_VanDerPol |
| 11 | EvoPINN11_LotkaVolterra |
| 12 | EvoPINN12_Schrodinger1D |

### ZDT Problems (5)

ZDT1, ZDT2, ZDT3, ZDT4, ZDT6

### DTLZ Problems (7)

DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7

### Mixed-Objective Problems (3)

Mixed_ZDT1, Mixed_DTLZ1, Mixed_DTLZ2

---

## Performance Metrics

MOBCO is evaluated using multiple complementary metrics that assess convergence, diversity, distribution, and computational efficiency.

| Metric | Description | Better when |
|---|---|---|
| **Hypervolume (HV)** | Volume of the objective space dominated by the obtained Pareto front relative to a reference point | Higher |
| **Inverted Generational Distance (IGD)** | Average distance between the obtained solution set and the reference Pareto-optimal front | Lower |
| **Generational Distance (GD)** | Distance between the obtained Pareto front and the reference Pareto front | Lower |
| **Spacing** | Uniformity of solution distribution along the Pareto front | Lower |
| **Spread** | Extent and distribution of solutions across the objective space | Depends on context |
| **Epsilon Indicator** | Minimum factor by which the obtained front must be translated to dominate the reference front | Lower |
| **Runtime** | Computational time required to complete optimization | Lower |

---

## Visualization

The framework includes several visualization tools for analyzing MOBCO's performance and behavior:

- **Expected vs. Obtained Pareto Front** — compares the theoretical Pareto front against MOBCO's obtained non-dominated solutions.
- **Objective-Space Plots** — 2D plots for two-objective problems, 3D plots for three-objective problems, and projections or parallel-coordinate plots for higher-dimensional problems.
- **Decision-Space Plots** — visualize candidate solutions in the original decision-variable space prior to objective mapping.
- **Convergence Plots** — track metrics such as Hypervolume, IGD, and GD across iterations to assess convergence stability.
- **Distribution Plots** — analyze the spread and uniformity of the final obtained Pareto solutions.

---

## Sensitivity Analysis

A global, variance-based **Sobol sensitivity analysis** is included to study the influence of MOBCO's control parameters on optimization performance, using **Hypervolume (HV)** as the primary response metric. Parameter combinations are generated via **Saltelli sampling**.

### Structural Parameters

- Population Size
- Archive Size
- Mutation Rate
- Number of Dogs
- Eyeing-Trigger Stagnation Length

### Dynamics-Shaping Parameters

- Acceleration Damping
- Mutation Decay Floor
- Step Decay Floor
- Stalking Pull
- Gathering Pull

### What the Analysis Reveals

- Parameters with the strongest influence on performance
- Parameters with relatively small effects
- First-order parameter effects
- Total-order parameter effects
- Parameter interaction effects

Run it with:

```bash
python sensitivity_analysis.py
```

---

## Statistical Analysis

The `Anova_independent_runs_with_box_plots/` module performs statistical comparison of MOBCO's combined-fitness results across independent runs using **ANOVA**, with automated **box-plot generation** for visual interpretation of result variance across benchmark problems.

---

## Related Work

Comparison algorithms used to benchmark MOBCO against other multi-objective optimizers are maintained in a separate repository:

🔗 **[Multi-Objective Border Collie Optimization — Comparison Algorithms](https://github.com/Samiksha-bajoria/Multi_objective_border_collie_optimization)**

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This work was carried out as part of a **Research Internship under IEEE CIS (Computational Intelligence Society)**, focused on the design, implementation, and empirical evaluation of Multi-Objective Border Collie Optimization.
