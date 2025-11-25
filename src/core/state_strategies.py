
from abc import ABC, abstractmethod
import numpy as np
from src.config import SimulationConfig

class StateProvider(ABC):
    """
    Abstract base class for defining how the state is computed in the SPGG simulation.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config

    @abstractmethod
    def get_state(self, R: np.ndarray, Sn: np.ndarray) -> np.ndarray:
        """
        Computes the state matrix based on reputation (R) and/or current strategies (Sn).
        
        Args:
            R: Reputation matrix (LxL).
            Sn: Current strategy matrix (LxL), where 0=Coop, 1=Defect.
            
        Returns:
            State matrix (LxL) with integer states.
        """
        pass

class ReputationStateProvider(StateProvider):
    """
    State is defined by the average reputation of neighbors (and potentially self).
    Corresponds to 'rep-state/main.py'.
    """
    def get_state(self, R: np.ndarray, Sn: np.ndarray) -> np.ndarray:
        L = self.config.L
        if self.config.use_second_order:
            offsets = [
                (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (2, 0), (-2, 0), (0, 2), (0, -2),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            ]
        else:
            offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
            
        sum_rep = np.zeros((L, L))
        for dx, dy in offsets:
            sum_rep += np.roll(R, shift=(dx, dy), axis=(0, 1))
            
        avg_rep = sum_rep / len(offsets)
        # State 1 if avg_rep > 0 (Good), State 0 otherwise (Bad)
        return (avg_rep > 0).astype(int)

class ActionStateProvider(StateProvider):
    """
    State is defined by the current action (strategy) of the agent.
    Corresponds to 'action_state/action.py'.
    """
    def get_state(self, R: np.ndarray, Sn: np.ndarray) -> np.ndarray:
        # State 0 if Cooperate (0), State 1 if Defect (1)
        # Assuming Sn is 0 for C and 1 for D.
        return (Sn == 0).astype(int) 
        # WAIT: In action.py: return (self._Sn == 0).astype(int)
        # If Sn=0 (Coop), returns 1. If Sn=1 (Defect), returns 0.
        # Let's double check action.py logic.
        # action.py line 108: return (self._Sn == 0).astype(int)
        # So State 1 = Cooperating, State 0 = Defecting.
        # This seems to be the mapping.
