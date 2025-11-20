"""SPGG Model Module"""
from .spgg import SPGG
from .algorithms import (
    RLAlgorithm, QLearning, SARSA, ExpectedSARSA, DoubleQLearning, create_algorithm
)

__all__ = ['SPGG', 'RLAlgorithm', 'QLearning', 'SARSA', 'ExpectedSARSA', 
           'DoubleQLearning', 'create_algorithm']

