from Problem import Problem
from src.solver import Solver


def solution(p: Problem):
    """
    Main entry point for the solver.
    
    Args:
        p: Problem instance
        
    Returns:
        List of tuples [(city, gold), ...] representing the solution path
    """
    # Initialize solver and find optimal solution
    solver = Solver(p, show_progress=False)
    solver.solve(show_progress=False)
    
    # Convert to required output format
    return solver.getSolution()
