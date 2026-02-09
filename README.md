# Exam project - Traveling gold collector

## Overview

This repository contains my solution for the **Exam project** of the course *Computational Intelligence (CI2025/26)*.
It addresses a variant of the **Traveling Salesman Problem** where the goal is to collect gold from cities while minimizing the total travel cost, considering that carrying gold increases transportation costs.

---

## Problem definition

The problem extends the classical TSP with a gold collection mechanism and dynamic travel costs.

* Each **city** (except city 0, the base) has:
  * A position in 2D space
  * A gold amount `gold_i` to be collected
* Each **edge** between cities has:
  * A distance `dist(i, j)` based on Euclidean distance
* The **cost function** for traveling from city `i` to city `j` while carrying weight `w` is:
  ```
  cost = dist(i,j) + (α × dist(i,j) × w)^β
  ```

The goal is to design one or more tours starting and ending at city 0 that:
* Visit all cities to collect their gold
* Minimize the **total travel cost**, considering the increasing cost of carrying gold
* Respect the graph connectivity (only existing edges can be used)

The parameters α, β, and density define different problem variants with varying difficulty levels.

---

## Representation of solutions

Solutions are represented as a list of tours, where each tour is a sequence of `(city, gold_amount)` tuples:

```python
tour = [(city_1, gold_1), (city_2, gold_2), ..., (city_k, gold_k)]
```

This representation allows:
* Multiple visits to the same city (useful when β > 1, where splitting gold collection can reduce costs)
* Partial gold collection per visit
* Flexible tour construction and manipulation

The final solution format includes `(0, 0)` markers between tours to indicate returns to base.

---

## Solution strategy

### 1. Distance oracle

For efficient distance lookups, a **DistanceOracle** class precomputes all-pairs shortest paths using Dijkstra's algorithm.
* For large graphs (n ≥ 200), all distances are precomputed upfront
* For smaller graphs, distances are computed lazily and cached
* This provides O(1) distance queries after initialization

---

### 2. Initialization

The algorithm employs **adaptive initialization strategies** based on problem parameters:

#### For β < 1 (cost decreases with weight):
* **Farthest-First initialization**: builds a single tour visiting cities in order of decreasing distance from base
* **Iterated 2-Opt**: applies 2-opt edge swaps iteratively until no improvement is found
* This approach works well because carrying more gold reduces cost, favoring single-tour solutions

#### For β = 1 (linear cost):
* **Greedy initialization**: iteratively selects the nearest unvisited city weighted by its gold amount
* Fast variant for large instances (n ≥ 200) uses randomized selection

#### For β > 1 (cost increases with weight):
* **Greedy partial gold collection**: builds multiple tours, collecting partial gold at each visit
* Limits carried gold to a threshold: `max_load = 500 / (α^(1/β))`
* Cities can be visited multiple times to avoid carrying excessive weight
* Fast variant for large instances uses randomized nearest-neighbor heuristic

---

### 3. Optimization via Simulated Annealing (SA)

After initialization, the solution is refined using **simulated annealing** with the following neighborhood operators:

#### Neighborhood operators:

1. **Iterated 2-Opt** (30% probability):
   * Reverses tour segments to eliminate crossing edges
   * Iterates until local optimum is reached
   
2. **Swap visits between tours** (25% probability):
   * Exchanges `(city, gold)` visits between two different tours
   
3. **Split tour** (15% probability):
   * Divides a tour into two separate tours to reduce accumulated weight
   
4. **Merge tours** (10% probability):
   * Combines two tours if it reduces total cost
   
5. **Split gold collection** (10% probability):
   * Splits a single visit into two visits collecting partial gold
   
6. **Relocate segment** (5% probability):
   * Moves a sequence of visits from one tour to another
   
7. **Random reassign** (5% probability):
   * Randomly reassigns a visit to a different tour for diversification

#### Annealing schedule:
* Initial temperature: `T₀ = 0.1 × initial_cost`
* Cooling rate: `T = T × 0.999` per iteration
* Acceptance probability: `P(accept) = exp(-Δcost / T)`
* **Early stopping**: terminates if no improvement for 1000 consecutive iterations

#### Adaptive iteration counts:
* β > 1, density > 0.7: 10,000 steps (since computing the initialization already takes long, I thought that taking some extra time to do more steps wouldn't be a bad idea in terms of time taken)
* β > 1, density ≤ 0.7: 7,500 steps
* β < 1: 2,000 steps (farthest-first + 2-opt already near-optimal)
* β = 1: 2,500 steps (since the baseline is already nearly optimal)

---

## Experimental results

The algorithm was benchmarked on 36 problem instances with varying parameters (N, β, density) with α fixed at 1.0. I decided to show what I thought would be relevant problem instances:
* N = 100, 550, 1000: to show a small, medium and high amount of cities
* β = 0.5, 1.0, 2.0, 5.0: to show how results vary with β < 1, β = 1, β > 1 and β >> 1
* density = 0.3, 0.6, 0.1: to show a small, medium and high value for density
Below are the detailed results showing **Improvement %** (cost reduction over baseline) and execution times (not including the time taken to compute the baseline's cost).

```
N      β    Density       Baseline          My cost          Improvement %    Time (s)  
---------------------------------------------------------------------------------------
100   0.5    0.3           1884.77          1391.65              26.2%        8.50
100   0.5    0.6           1464.27          1300.09              11.2%       19.36
100   0.5    1.0           1292.63          1276.98               1.2%       17.64
100   1.0    0.3          21007.50         21006.43               0.0%       13.62
100   1.0    0.6          18618.12         18617.41               0.0%       24.06
100   1.0    1.0          18266.19         18265.80               0.0%       38.09
100   2.0    0.3        4403745.42       1118615.18              74.6%       28.00
100   2.0    0.6        4860428.15        939793.93              80.7%       31.55
100   2.0    1.0        5404978.09        748419.38              86.2%       46.39
100   5.0    0.3          1.68e+14         1.34e+12              99.2%       26.66
100   5.0    0.6          2.52e+14         3.71e+12              98.5%       27.48
100   5.0    1.0          3.48e+14         1.42e+12              99.6%       45.63
550   0.5    0.3           9874.06          7538.65              23.7%        9.78
550   0.5    0.6           8575.81          7470.39              12.9%       11.01
550   0.5    1.0           7459.64          7443.75               0.2%       10.22
550   1.0    0.3         108273.81        108273.48               0.0%        3.36
550   1.0    0.6         107009.18        107008.26               0.0%        6.50
550   1.0    1.0         106544.96        106543.95               0.0%        6.25
550   2.0    0.3          2.22e+07         1.03e+07              53.8%       62.56
550   2.0    0.6          2.65e+07       9966496.08              62.4%       61.72
550   2.0    1.0          3.12e+07       8911755.47              71.4%       93.72
550   5.0    0.3          8.30e+14         5.01e+13              94.0%       61.76
550   5.0    0.6          1.28e+15         5.56e+13              95.7%       60.70
550   5.0    1.0          1.88e+15         4.09e+13              97.8%       93.97
1000  0.5    0.3          17573.08         13527.20              23.0%       19.06
1000  0.5    0.6          15270.40         13455.18              11.9%       16.75
1000  0.5    1.0          13456.96         13434.23               0.2%       19.93
1000  1.0    0.3         194204.60        194204.23               0.0%        7.05
1000  1.0    0.6         193015.66        193014.83               0.0%        8.43
1000  1.0    1.0         192936.23        192935.40               0.0%       13.83
1000  2.0    0.3          4.01e+07         2.17e+07              45.8%       99.60
1000  2.0    0.6          4.97e+07         2.23e+07              55.2%       99.35
1000  2.0    1.0          5.76e+07         2.06e+07              64.3%      155.16
1000  5.0    0.3          1.67e+15         1.50e+14              91.0%      105.18
1000  5.0    0.6          2.79e+15         1.49e+14              94.7%      108.09
1000  5.0    1.0          3.80e+15         1.35e+14              96.4%      151.00
```

---

## Performance analysis

### Key observations:

1. **β = 1.0**: greedy initialization + SA quickly converges, without being able to overcome baseline. With linear costs there's not much room for improvement.

2. **β < 1.0** (decreasing cost): 
   * Significant improvements (11-26%) over baseline
   * My strategy performs very well, showing all the benefits of starting from the farthest cities, since cost decreases with weight
   * Lower density benefits more from optimization

3. **β > 1.0** (increasing cost):
   * Dramatic improvements over naive baseline (45-99%)
   * Partial gold collection and multiple tours are essential
   * Higher β values show exponentially better improvements (99%+ for β=5)

4. **Scalability**: 
   * Fast initializers maintain good performance for N=1000
   * Execution times scale reasonably (under 3 minutes for largest instances)
   * Distance oracle precomputation pays off for large graphs

---

## Code structure

```
project-work/
├── s336732.py              # Main entry point
├── Problem.py              # Problem definition and cost functions
├── src/
│   ├── solver.py           # Complete solver implementation
│   │   ├── DistanceOracle
│   │   ├── Initialization strategies (Greedy, Farthest-First, etc.)
│   │   ├── Neighborhood operators (2-opt, swap, merge, etc.)
│   │   └── SimulatedAnnealing
│   └── __init__.py
└── README.md               # This file
```

---

**Author:** [Riccardo Dattena - s336732]  
**Course:** Computational Intelligence (CI2025/26)  
**Exam project — Traveling gold collector problem**