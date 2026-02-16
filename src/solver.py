import copy
import random
import math
import numpy as np
import networkx as nx
from tqdm.auto import tqdm

from Problem import Problem

# <<< DistanceOracle >>>
# Caches and provides efficient distance lookups between graph nodes

class DistanceOracle:
    def __init__(self, graph: nx.Graph, show_progress=False):
        '''
        Initialize distance oracle with optional precomputation
        - graph: NetworkX graph with 'dist' edge weights
        - show_progress: whether to display progress bar
        '''
        self.graph = graph
        self.cache = {}  # Stores distances
        self.paths = {}  # Stores shortest paths
        
        # For large graphs (n >= 200), pre-compute all distances to avoid lazy computation overhead
        n = len(graph.nodes)
        if n >= 200:
            self.precomputeAllDistances(show_progress)

    def precomputeAllDistances(self, show_progress=False):
        '''
        Pre-compute all pairwise shortest path distances and paths
        - show_progress: whether to display progress bar
        '''
        if show_progress:
            from tqdm import tqdm
            print("Pre-computing distance matrix and paths...")
            nodes = list(self.graph.nodes)
            for node in tqdm(nodes, desc="Distance Oracle"):
                lengths, paths = nx.single_source_dijkstra(self.graph, source=node, weight="dist")
                self.cache[node] = lengths
                self.paths[node] = paths
        else:
            nodes = list(self.graph.nodes)
            for node in nodes:
                lengths, paths = nx.single_source_dijkstra(self.graph, source=node, weight="dist")
                self.cache[node] = lengths
                self.paths[node] = paths

    def dist(self, i: int, j: int) -> float:
        '''
        Get shortest distance between two nodes
        - i: source node ID
        - j: target node ID
        - returns: shortest path distance
        '''
        if i == j:
            return 0.0
        if i not in self.cache:
            lengths, paths = nx.single_source_dijkstra(self.graph, source=i, weight="dist")
            self.cache[i] = lengths
            self.paths[i] = paths
        return self.cache[i][j]
    
    def path(self, i: int, j: int) -> list[int]:
        '''
        Get shortest path between two nodes
        - i: source node ID
        - j: target node ID
        - returns: list of nodes in shortest path from i to j
        '''
        if i == j:
            return [i]
        if i not in self.paths:
            lengths, paths = nx.single_source_dijkstra(self.graph, source=i, weight="dist")
            self.cache[i] = lengths
            self.paths[i] = paths
        return self.paths[i][j]


# <<< Solution >>>
# Represents a solution as a collection of tours with associated cost

class Solution:
    def __init__(self, tours: list[list[tuple[int, float]]]):
        '''
        Initialize solution with tours
        - tours: list of tours, each tour is a list of (city, gold_amount) tuples
        Note: cities can appear multiple times, gold_amount is collected in that visit!
        '''
        self.tours = tours
        self.cost = None


# *** COST FUNCTIONS ***

def solutionCost(
    sol: Solution,
    problem: Problem,
    oracle: DistanceOracle,
) -> float:
    '''
    Calculate total cost of a solution
    - sol: solution to evaluate
    - problem: problem instance
    - oracle: distance oracle for lookups
    - returns: total solution cost
    '''
    total = 0.0

    for tour in sol.tours:
        current = 0
        gold = 0.0

        for city, gold_amount in tour:
            d = oracle.dist(current, city)
            total += d + (problem.alpha * d * gold) ** problem.beta
            gold += gold_amount
            current = city

        # Return to base
        d = oracle.dist(current, 0)
        total += d + (problem.alpha * d * gold) ** problem.beta

    return total


def tourCost(tour, problem, oracle):
    '''
    Calculate cost of a single tour
    - tour: list of (city, gold_amount) tuples
    - problem: problem instance
    - oracle: distance oracle
    - returns: tour cost
    '''
    current = 0
    gold = 0.0
    total = 0.0

    for city, gold_amount in tour:
        d = oracle.dist(current, city)
        total += d + (problem.alpha * d * gold) ** problem.beta
        gold += gold_amount
        current = city

    d = oracle.dist(current, 0)
    total += d + (problem.alpha * d * gold) ** problem.beta
    return total


# *** INITIALIZERS ***

# <<< GreedyInitializer >>>
# Builds initial solution using greedy heuristic based on distance and gold

class GreedyInitializer:
    
    def __init__(self, problem: Problem, oracle: DistanceOracle):
        '''
        Initialize greedy builder
        - problem: problem instance
        - oracle: distance oracle
        '''
        self.problem = problem
        self.oracle = oracle

    def build(self, show_progress=False) -> Solution:
        '''
        Build initial solution greedily
        - show_progress: whether to display progress bar
        - returns: initial solution
        '''
        from tqdm import tqdm
        unvisited = set(self.problem.graph.nodes)
        unvisited.remove(0)

        tours = []
        total_cities = len(unvisited)
        
        iterator = tqdm(total=total_cities, desc="Building initial solution", disable=not show_progress)

        while unvisited:
            tour = []
            current = 0
            gold = 0.0

            while unvisited:
                def score(c):
                    d = self.oracle.dist(current, c)
                    g = self.problem.graph.nodes[c]["gold"]
                    return d * (g ** (self.problem.beta - 1))

                city = min(unvisited, key=score)

                # Take all gold from the city
                city_gold = self.problem.graph.nodes[city]["gold"]
                tour.append((city, city_gold))
                gold += city_gold
                current = city
                unvisited.remove(city)
                iterator.update(1)

            tours.append(tour)
        
        iterator.close()
        return Solution(tours)


# <<< GreedyInitializerWithPartialGold >>>
# Builds solution allowing partial gold collection per visit (used for beta > 1)

class GreedyInitializerWithPartialGold:
    
    def __init__(self, problem: Problem, oracle: DistanceOracle):
        '''
        Initialize partial gold collector
        - problem: problem instance
        - oracle: distance oracle
        '''
        self.problem = problem
        self.oracle = oracle

    def build(self, show_progress=False) -> Solution:
        '''
        Build solution with partial gold collection strategy
        - show_progress: whether to display progress bar
        - returns: initial solution
        '''
        from tqdm import tqdm
        # Map city -> remaining gold
        remaining_gold = {
            city: self.problem.graph.nodes[city]["gold"]
            for city in self.problem.graph.nodes
            if city != 0
        }

        tours = []
        total_gold = sum(remaining_gold.values())
        processed_gold = 0.0
        
        pbar = tqdm(total=100, desc="Building initial solution", disable=not show_progress, unit="%")

        while any(g > 0.01 for g in remaining_gold.values()):
            tour = []
            current = 0
            carried_gold = 0.0
            
            # Max load threshold for beta > 1
            if self.problem.beta > 1:
                max_load = 500 / (self.problem.alpha ** (1/self.problem.beta))
            else:
                max_load = float('inf')

            while True:
                # Find cities not fully visited
                unvisited = {c: g for c, g in remaining_gold.items() if g > 0.01}
                if not unvisited:
                    break

                # Heuristic: minimize weighted distance by potential gold
                def score(c):
                    d = self.oracle.dist(current, c)
                    gold_potential = remaining_gold[c]
                    if self.problem.beta > 1:
                        # For beta > 1, favor nearby cities with little gold
                        return d / (gold_potential ** 0.5 + 0.1)
                    else:
                        return d / (gold_potential + 0.1)

                city = min(unvisited, key=score)
                
                # Decide how much gold to take
                available = remaining_gold[city]
                
                if self.problem.beta > 1:
                    # Take only what max_load allows
                    can_carry = max(0, max_load - carried_gold)
                    take_gold = min(available, can_carry)
                    
                    if take_gold < 0.01:
                        # Full load, end this tour
                        break
                else:
                    # beta == 1: take everything
                    take_gold = available

                tour.append((city, take_gold))
                remaining_gold[city] -= take_gold
                carried_gold += take_gold
                current = city
                
                # Update progress bar
                if show_progress:
                    processed_gold += take_gold
                    pbar.n = int((processed_gold / total_gold) * 100)
                    pbar.refresh()

                # If beta > 1 and load near max, return to base
                if self.problem.beta > 1 and carried_gold >= max_load * 0.9:
                    break

            if tour:
                tours.append(tour)
            else:
                # Avoid infinite loop
                break
        
        pbar.close()
        return Solution(tours)


# *** FAST INITIALIZATION UTILITIES ***

def buildGoldDict(graph: nx.Graph):
    '''
    Build O(1) lookup dictionary for node gold amounts
    - graph: NetworkX graph
    - returns: gold lookup dictionary
    '''
    return {node: graph.nodes[node]['gold'] for node in graph.nodes if node != 0}


# <<< FastGreedyInitializer >>>
# Fast random initialization for large problem instances

class FastGreedyInitializer:
    def __init__(self, problem: Problem, oracle: DistanceOracle):
        '''
        Initialize fast builder with precomputed data
        - problem: problem instance
        - oracle: distance oracle
        '''
        self.problem = problem
        self.oracle = oracle
        self.gold_dict = buildGoldDict(problem.graph)
    
    def build(self, show_progress=False) -> Solution:
        from tqdm import tqdm
        unvisited = set(self.problem.graph.nodes)
        unvisited.remove(0)
        
        tour = []
        
        iterator = tqdm(total=len(unvisited), desc="Building initial solution", disable=not show_progress)
        
        # Random traversal using pre-computed shortest paths
        while unvisited:
            # Pick random unvisited city
            city = random.choice(list(unvisited))
            
            # Get gold amount
            city_gold = self.gold_dict[city]
            tour.append((city, city_gold))
            unvisited.remove(city)
            iterator.update(1)
        
        iterator.close()
        return Solution([tour])


# <<< FastGreedyInitializerWithPartialGold >>>
# Fast initialization with partial gold collection for large instances

class FastGreedyInitializerWithPartialGold:
    def __init__(self, problem: Problem, oracle: DistanceOracle):
        '''
        Initialize fast partial gold collector
        - problem: problem instance
        - oracle: distance oracle
        '''
        self.problem = problem
        self.oracle = oracle
        self.gold_dict = buildGoldDict(problem.graph)
    
    def build(self, show_progress=False) -> Solution:
        from tqdm import tqdm
        # Track remaining gold per city
        remaining_gold = self.gold_dict.copy()
        
        tours = []
        total_gold = sum(remaining_gold.values())
        processed_gold = 0.0
        
        pbar = tqdm(total=100, desc="Building initial solution", disable=not show_progress, unit="%")
        
        while any(g > 0.01 for g in remaining_gold.values()):
            tour = []
            current = 0
            carried_gold = 0.0
            
            # Max load threshold for beta > 1
            if self.problem.beta > 1:
                max_load = 500 / (self.problem.alpha ** (1/self.problem.beta))
            else:
                max_load = float('inf')
            
            while True:
                # Find unvisited cities with remaining gold
                unvisited = [c for c, g in remaining_gold.items() if g > 0.01]
                if not unvisited:
                    break
                
                # Pick nearest city using oracle (greedy for small sets)
                if len(unvisited) <= 20:
                    # For small sets, use greedy
                    city = min(unvisited, key=lambda c: self.oracle.dist(current, c))
                else:
                    # For large sets, pick randomly from nearest 20
                    distances = [(c, self.oracle.dist(current, c)) for c in random.sample(unvisited, min(20, len(unvisited)))]
                    distances.sort(key=lambda x: x[1])
                    city = distances[0][0]
                
                # Decide how much gold to take
                available = remaining_gold[city]
                
                if self.problem.beta > 1:
                    can_carry = max(0, max_load - carried_gold)
                    take_gold = min(available, can_carry)
                    
                    if take_gold < 0.01:
                        break
                else:
                    take_gold = available
                
                tour.append((city, take_gold))
                remaining_gold[city] -= take_gold
                carried_gold += take_gold
                current = city
                
                # Update progress bar
                if show_progress:
                    processed_gold += take_gold
                    pbar.n = int((processed_gold / total_gold) * 100)
                    pbar.refresh()
                
                # Return to base if near max load
                if self.problem.beta > 1 and carried_gold >= max_load * 0.9:
                    break
            
            if tour:
                tours.append(tour)
            else:
                break
        
        pbar.close()
        return Solution(tours)


# <<< FarthestFirstInitializer >>>
# Builds tour starting from farthest cities (good for beta < 1)

class FarthestFirstInitializer:
    def __init__(self, problem: Problem, oracle: DistanceOracle):
        '''
        Initialize farthest-first builder
        - problem: problem instance
        - oracle: distance oracle
        '''
        self.problem = problem
        self.oracle = oracle
        self.gold_dict = buildGoldDict(problem.graph)
    
    def build(self, show_progress=False) -> Solution:
        from tqdm import tqdm
        
        # Get all cities and their distances from depot
        cities = list(self.gold_dict.keys())
        city_distances = [(city, self.oracle.dist(0, city)) for city in cities]
        
        # Sort by distance from depot (descending - farthest first)
        city_distances.sort(key=lambda x: x[1], reverse=True)
        
        # Build tour visiting farthest cities first
        tour = []
        for city, _ in city_distances:
            city_gold = self.gold_dict[city]
            tour.append((city, city_gold))
        
        if show_progress:
            print(f"Built farthest-first tour with {len(tour)} cities")
        
        return Solution([tour])


def iterated2Opt(sol: Solution, problem: Problem, oracle: DistanceOracle, max_iterations=1000, show_progress=False) -> Solution:
    '''
    Apply 2-opt local search iteratively until no improvement
    - sol: initial solution
    - problem: problem instance
    - oracle: distance oracle
    - max_iterations: maximum iterations
    - show_progress: whether to display progress bar
    - returns: optimized solution
    '''
    if len(sol.tours) != 1:
        return sol  # Only works for single tour
    
    tour = sol.tours[0]
    n = len(tour)
    
    if n < 4:
        return sol
    
    best_cost = tourCost(tour, problem, oracle)
    improved = True
    iteration = 0
    
    iterator = tqdm(desc="2-opt optimization", disable=not show_progress, total=max_iterations)
    
    while improved and iteration < max_iterations:
        improved = False
        
        for i in range(n - 2):
            for j in range(i + 2, n):
                # Try reversing segment [i+1:j+1]
                new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                new_cost = tourCost(new_tour, problem, oracle)
                
                if new_cost < best_cost:
                    tour = new_tour
                    best_cost = new_cost
                    improved = True
                    if show_progress:
                        iterator.set_postfix({"cost": f"{best_cost:.2f}"})
                    break
            
            if improved:
                break
        
        iteration += 1
        iterator.update(1)
    
    iterator.close()
    
    if show_progress:
        print(f"2-opt completed in {iteration} iterations")
    
    return Solution([tour])


# *** OPERATORS ***

def swapInTour(sol: Solution) -> Solution:
    '''
    Swap two random visits within a tour
    - sol: input solution
    - returns: modified solution
    '''
    sol = copy.deepcopy(sol)
    tour = random.choice(sol.tours)
    if len(tour) < 2:
        return sol

    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]
    return sol


def twoOpt(sol: Solution) -> Solution:
    '''
    Apply 2-opt move: reverse a segment of a tour
    - sol: input solution
    - returns: modified solution
    '''
    sol = copy.deepcopy(sol)
    tour = random.choice(sol.tours)
    if len(tour) < 4:
        return sol

    i, j = sorted(random.sample(range(len(tour)), 2))
    tour[i:j] = reversed(tour[i:j])
    return sol


def moveCityVisit(sol: Solution) -> Solution:
    '''
    Move a visit from one tour to another
    - sol: input solution
    - returns: modified solution
    '''
    sol = copy.deepcopy(sol)
    if len(sol.tours) < 2:
        return sol

    t1, t2 = random.sample(sol.tours, 2)
    if not t1:
        return sol

    visit = random.choice(t1)
    t1.remove(visit)
    t2.insert(random.randint(0, len(t2)), visit)

    if not t1:
        sol.tours.remove(t1)

    return sol


def splitTour(sol: Solution) -> Solution:
    '''
    Split a tour into two separate tours
    - sol: input solution
    - returns: modified solution
    '''
    sol = copy.deepcopy(sol)
    tour = random.choice(sol.tours)
    if len(tour) < 2:
        return sol

    k = random.randint(1, len(tour) - 1)
    t1 = tour[:k]
    t2 = tour[k:]

    sol.tours.remove(tour)
    sol.tours.append(t1)
    sol.tours.append(t2)

    return sol


def smartMerge(sol, problem, oracle):
    '''
    Merge two tours if it improves total cost
    - sol: input solution
    - problem: problem instance
    - oracle: distance oracle
    - returns: modified solution
    '''
    sol = copy.deepcopy(sol)

    if len(sol.tours) < 2:
        return sol

    t1, t2 = random.sample(sol.tours, 2)

    c_before = tourCost(t1, problem, oracle) + tourCost(t2, problem, oracle)
    merged = t1 + t2
    c_after = tourCost(merged, problem, oracle)

    if c_after < c_before:
        sol.tours.remove(t1)
        sol.tours.remove(t2)
        sol.tours.append(merged)

    return sol


def splitGoldCollection(sol: Solution) -> Solution:
    '''
    Split gold collection at a city into two separate visits
    - sol: input solution
    - returns: modified solution
    '''
    sol = copy.deepcopy(sol)
    
    tour = random.choice(sol.tours)
    if not tour:
        return sol
    
    # Find visits with enough gold to split
    candidates = [i for i, (city, gold) in enumerate(tour) if gold > 10]
    if not candidates:
        return sol
    
    idx = random.choice(candidates)
    city, gold = tour[idx]
    
    # Split gold into two parts (between 30% and 70%)
    split_ratio = 0.3 + random.random() * 0.4
    gold1 = gold * split_ratio
    gold2 = gold * (1 - split_ratio)
    
    # Create new tour with the second part
    new_tour = [(city, gold2)]
    tour[idx] = (city, gold1)
    sol.tours.append(new_tour)
    
    return sol


def mergeGoldVisits(sol: Solution) -> Solution:
    '''
    Merge multiple visits to same city into one visit
    - sol: input solution
    - returns: modified solution
    '''
    sol = copy.deepcopy(sol)
    
    # Find all visits per city
    city_visits = {}
    for tour_idx, tour in enumerate(sol.tours):
        for visit_idx, (city, gold) in enumerate(tour):
            if city not in city_visits:
                city_visits[city] = []
            city_visits[city].append((tour_idx, visit_idx, gold))
    
    # Find cities with multiple visits
    multi_visit_cities = [c for c, visits in city_visits.items() if len(visits) > 1]
    if not multi_visit_cities:
        return sol
    
    city = random.choice(multi_visit_cities)
    visits = city_visits[city]
    
    # Choose two visits to merge
    if len(visits) < 2:
        return sol
    
    (t1_idx, v1_idx, g1), (t2_idx, v2_idx, g2) = random.sample(visits, 2)
    
    # Remove second visit and add its gold to the first
    tour1 = sol.tours[t1_idx]
    tour2 = sol.tours[t2_idx]
    
    tour1[v1_idx] = (city, g1 + g2)
    del tour2[v2_idx]
    
    if not tour2:
        sol.tours.remove(tour2)
    
    return sol


def randomReassign(sol: Solution) -> Solution:
    '''
    Randomly reassign all visits to new tours
    - sol: input solution
    - returns: modified solution
    '''
    sol = copy.deepcopy(sol)
    all_visits = [(c, g) for t in sol.tours for c, g in t]
    random.shuffle(all_visits)
    sol.tours = [[(c, g)] for c, g in all_visits]
    return sol


def randomNeighbor(self, sol):
    '''
    Generate random neighbor solution using appropriate operators
    - sol: current solution
    - returns: neighbor solution
    '''
    ops = [twoOpt, swapInTour]

    if self.problem.beta == 1:
        ops += [
            lambda s: smartMerge(s, self.problem, self.oracle), 
            lambda s: mergeGoldVisits(s)
        ]
    elif self.problem.beta > 1:
        ops += [
            moveCityVisit, 
            splitTour, 
            lambda s: splitGoldCollection(s)
        ]
        # Favor operators that create shorter trips
        if random.random() < 0.3:
            ops.append(lambda s: splitGoldCollection(s))
    else:  # beta < 1
        # For beta < 1, favor long tours with consolidation
        ops += [
            lambda s: smartMerge(s, self.problem, self.oracle),
            lambda s: mergeGoldVisits(s)
        ]
        # Favor merging to create longer tours
        if random.random() < 0.4:
            ops.append(lambda s: smartMerge(s, self.problem, self.oracle))

    if random.random() < 0.05:
        ops.append(lambda s: randomReassign(s))

    return random.choice(ops)(sol)



# <<< SimulatedAnnealing >>>
# Metaheuristic optimizer using simulated annealing or hill climbing

class SimulatedAnnealing:
    def __init__(self, problem, oracle, use_annealing=True):
        '''
        Initialize optimizer
        - problem: problem instance
        - oracle: distance oracle
        - use_annealing: if False, acts as hill climbing
        '''
        self.problem = problem
        self.oracle = oracle
        self.use_annealing = use_annealing

    def run(self, sol: Solution, steps=1000, show_progress=False, patience=1000):
        '''
        Run optimization process
        - sol: initial solution
        - steps: maximum iterations
        - show_progress: whether to display progress bar
        - patience: early stopping threshold
        - returns: best solution found
        '''
        best = sol
        best.cost = solutionCost(sol, self.problem, self.oracle)
        current = best
        T = best.cost * 0.1

        # Early stopping
        no_improvement_count = 0
        best_cost_so_far = best.cost

        iterator = tqdm(range(steps), desc="Optimizing", disable=not show_progress)
        
        for step in iterator:
            neighbor = self.randomNeighbor(current)
            neighbor.cost = solutionCost(neighbor, self.problem, self.oracle)

            delta = neighbor.cost - current.cost

            # If use_annealing=False, becomes Hill Climbing (accept only improvements)
            if self.use_annealing:
                accept = delta < 0 or random.random() < math.exp(-delta / T)
            else:
                accept = delta < 0
            
            if accept:
                current = neighbor

                if current.cost < best.cost:
                    best = current
                    
                    # Improvement found, reset counter
                    if current.cost < best_cost_so_far:
                        best_cost_so_far = current.cost
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1

            T *= 0.999
            
            # Update progress bar with current cost
            if show_progress:
                iterator.set_postfix({
                    "best_cost": f"{best.cost:.2f}", 
                    "T": f"{T:.2f}",
                    "patience": f"{no_improvement_count}/{patience}"
                })
            
            # Early stopping
            if no_improvement_count >= patience:
                if show_progress:
                    print(f"\nEarly stopping: no improvement for {patience} iterations")
                break

        return best


SimulatedAnnealing.randomNeighbor = randomNeighbor



# <<< Solver >>>
# Main solver class that coordinates initialization and optimization

class Solver:
    def __init__(self, problem: Problem, show_progress=False):
        '''
        Initialize solver with problem instance
        - problem: problem instance to solve
        - show_progress: whether to display progress during initialization
        '''
        self.problem = problem
        self.oracle = DistanceOracle(problem.graph, show_progress=show_progress)

    def solve(self, show_progress=False):
        '''
        Solve the problem using appropriate strategy based on beta and problem size
        - show_progress: whether to display progress during solving
        '''
        # Choose fast initializers for large instances (n >= 200)
        n = len(self.problem.graph.nodes)
        use_fast = n >= 200
        
        # Use appropriate initializer based on beta and problem size
        if self.problem.beta > 1:
            if use_fast:
                init = FastGreedyInitializerWithPartialGold(self.problem, self.oracle).build(show_progress=show_progress)
            else:
                init = GreedyInitializerWithPartialGold(self.problem, self.oracle).build(show_progress=show_progress)
        elif self.problem.beta < 1:
            # For beta < 1: use farthest-first + 2-opt (always best performer)
            init = FarthestFirstInitializer(self.problem, self.oracle).build(show_progress=show_progress)
            init = iterated2Opt(init, self.problem, self.oracle, max_iterations=500, show_progress=show_progress)
        else:  # beta == 1
            if use_fast:
                init = FastGreedyInitializer(self.problem, self.oracle).build(show_progress=show_progress)
            else:
                init = GreedyInitializer(self.problem, self.oracle).build(show_progress=show_progress)
            
        init.cost = solutionCost(init, self.problem, self.oracle)

        # Simulated Annealing with more iterations for complex problems
        # For beta < 1, reduce SA steps since farthest-first + 2-opt already optimizes
        sa = SimulatedAnnealing(self.problem, self.oracle)
        if self.problem.beta > 1 and self.problem.density > 0.7:
            steps = 7500 # changed at last from 10k to 7.5k to save time
        elif self.problem.beta > 1 and self.problem.density <= 0.7:
            steps = 7500
        elif self.problem.beta < 1:
            steps = 2000
        else:
            steps = 2500
        best = sa.run(init, steps=steps, show_progress=show_progress)

        self.best = best

    def getSolution(self):
        '''
        Get solution in required format with expanded shortest paths
        - returns: list of (city, cumulative_gold) tuples representing valid path in graph
        '''
        out = []
        
        for tour in self.best.tours:
            current_city = 0
            cumulative_gold = 0.0
            
            for target_city, gold_amount in tour:
                # Get shortest path from current to target city
                path = self.oracle.path(current_city, target_city)
                
                # Add all intermediate cities in the path (excluding starting city, including target)
                for node in path[1:]:
                    # If this is the target city, add the gold
                    if node == target_city:
                        cumulative_gold += gold_amount
                        out.append((node, cumulative_gold))
                    else:
                        # Intermediate city - no gold collected
                        out.append((node, cumulative_gold))
                
                current_city = target_city
            
            # Return to base from last city
            return_path = self.oracle.path(current_city, 0)
            for node in return_path[1:]:
                if node == 0:
                    # Unload gold at base
                    out.append((0, 0))
                else:
                    # Intermediate city - still carrying gold
                    out.append((node, cumulative_gold))
        
        return out
