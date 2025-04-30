import numpy as np
from typing import List
from pymoo.core.sampling import Sampling

from search_space.cgp.cell import Cell


class SamplingCGP(Sampling):

    def __init__(self, configuration, reduction, N, R, weights):
        """
        This abstract class represents any sampling strategy that can be used to create an initial population or
        an initial search point.
        """
        super().__init__()

        self.configuration = configuration
        self.reduction_config = reduction
        self.normal = N
        self.reduce = R
        self.weights = weights


    def _do(self, problem, n_samples, **kwargs):
        # Create an array to store sampled CGP individuals
        X = np.empty((n_samples, 1), dtype=object)

        # Populate each individual in the array
        for n in range(n_samples):
            X[n, 0] = Cell(self.configuration, self.reduction_config, self.normal, self.reduce, self.weights)
        return X


