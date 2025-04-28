from pymoo.core.sampling import Sampling
from .CellCGPb import CellCgpb
from typing import List
import numpy as np


class CartesianCellGeneticProgrammingB(Sampling):
    """
    Sampling class for generating individuals in Cartesian Genetic Programming (CGP).

    This class is designed to create CGP structures based on given configurations
    (conf_net), pooling, and dimensions (N, R).

    Attributes:
        conf_net (list): List of configurations for each CGP layer.
        Mpool (list): Pool of model configurations for reduction blocks.
        N (int): Number of normal blocks in the architecture.
        R (int): Number of reduction blocks between normal blocks.
    """

    def __init__(self,
                 conf_net: List = None,
                 N: int = 1):
        """
        Initializes the CGP sampling configuration.

        Args:
            conf_net (list): List of configurations for each CGP layer.
            pool (list): Pool of configurations for reduction blocks.
            N (int): Number of normal blocks in the architecture.
            R (int): Number of reduction blocks between normal blocks.
        """
        super().__init__()
        self.N = N
        self.conf_net = conf_net

    def _do(self, problem, n_samples, **kwargs):
        """
        Generates the initial population of CGP individuals for the algorithm.

        Args:
            problem: The optimization problem instance (not used in this method).
            n_samples (int): Number of individuals to sample.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: Array of CGP individuals as initial population.
        """
        # Create an array to store sampled CGP individuals
        X = np.full((n_samples, 1), None, dtype=object)

        # Populate each individual in the array
        for i in range(n_samples):
            gen = CellCgpb(self.conf_net, self.N)
            X[i, 0] = gen  # Assign generated individual
        return X

    def __testpop__(self, n_samples):
        """
        Generates a test population of CGP individuals without involving the problem instance.

        Args:
            n_samples (int): Number of individuals to sample.

        Returns:
            list: List of CGP individuals.
        """
        # Create a list to store the test population
        test_population = []

        # Populate the test population
        for i in range(n_samples):
            gen = CellCgpb(self.conf_net, self.N)
            test_population.append(gen)  # Append generated individual to list
        return test_population