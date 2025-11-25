
import numpy as np

def overlap5(A: np.ndarray) -> np.ndarray:
    """
    Computes the sum of a position and its four neighbors (von Neumann neighborhood).
    Periodic boundary conditions are applied (toroidal grid).
    
    Args:
        A: Input 2D array.
        
    Returns:
        2D array where each cell is the sum of itself and its 4 neighbors.
    """
    return (A + 
            np.roll(A, -1, axis=0) + 
            np.roll(A, 1, axis=0) + 
            np.roll(A, -1, axis=1) + 
            np.roll(A, 1, axis=1))
