from __future__ import annotations

from pymoo.algorithms.moo.nsga2 import NSGA2
from typing import Dict

from strategy.cgpnas.CrossoverCellCgpW import CrossoverCellCgpW
from strategy.cgpnas.MutationCellCgpW import MutationCellCgpW


class CGPNASV2():
    def __init__(self,
                 population_size: int = 100,
                 generations: int = 10,
                 mutation_prob: float = 0.3,
                 mutation_eta: int = 15,
                 crossover_prob: float = 0.9,
                 crossover_eta: int = 20,
                 dedup: bool = False,
                 seed: int = None,
                 callback: Dict = None
                 ):
        self.population_size = population_size
        self.generations = generations
        self.crossover = crossover_prob
        self.crossover_eta = crossover_eta
        self.mutation = mutation_prob
        self.mutation_eta = mutation_eta
        self.dedup = dedup
        self.seed = seed

    def define_algorithm(self, search_space, callback=None):
        return NSGA2(
            pop_size=self.population_size,
            sampling=search_space,
            crossover=CrossoverCellCgpW(prob=self.crossover, eta=self.crossover_eta),
            mutation=MutationCellCgpW(prob=self.mutation, eta=self.mutation_eta),
            eliminate_duplicates=self.dedup,
            callback=callback,
            seed=self.seed
        )
