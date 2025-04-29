import copy
import os
import pickle
from copy import deepcopy

import matplotlib
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga2 import RankAndCrowding
from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.indicators.hv import Hypervolume
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import ZeroToOneNormalization

from strategy.surrogate.models.carts import CART
from strategy.surrogate.subset_selection import subset_selection

matplotlib.use('Agg')


class SAMOS(Algorithm):
    def __init__(self,
                 sample_space: Sampling,  # Define search space
                 problem: Problem = None,  # Define search problem
                 n_doe: int = 100,  # Nr of initial architectures to fit surrogate
                 n_infill: int = 8,  # Nr of architectures to train
                 n_gen_candidates: int = 20,  # Nr of generations of NSGA-II loop
                 n_gen_surrogate: int = 30,  # Nr of generations of the surrogate model
                 n_var: int = None,  # Nr of variables
                 logger_params=None,
                 sbatch='',
                 use_archive=None,
                 continue_id=None,
                 sa_algorithm=None,  # Genetic algorithm parameters
                 verbose=1, # Verbose 0: silent, Verbose 1: Images, Verbose 2: Text only
                 **kwargs):

        super().__init__(eliminate_duplicates=False, **kwargs)
        self.sample_space = sample_space
        self.problem = problem
        self.logger_params = logger_params
        self.sa_algorithm = sa_algorithm
        self.sbatch = sbatch
        self.use_archive = use_archive
        self.continue_id = continue_id
        self.verbose = verbose

        self.save_dir = os.path.join(self.logger_params['save_dir'],
                                     self.logger_params['name'],
                                     self.logger_params['version'],
                                     'archive')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.n_doe = n_doe
        self.n_infill = n_infill
        self.n_gen_candidates = n_gen_candidates
        self.n_gen_surrogate = n_gen_surrogate
        self.n_var = n_var
        self.it = 0

        self.surrogate = None
        self.predictions = np.zeros((self.n_gen_surrogate + 1, self.n_infill))

        self.initialization = Initialization(sample_space)
        self._archive = Population()  # all solutions that have been evaluated so far
        self.infills = None  # here always the most recent infill solutions are stored

    def _setup(self, problem, **kwargs):
        # initialize surrogate predictor
        if self.surrogate is None:
            self.surrogate = CART(n_tree=500)

    def _initialize_infill(self):
        if self.continue_id is not None:
            if self.continue_id[0] is not None:
                with open(self.continue_id[0], "rb") as f:
                    self.infills = pickle.load(f)
                    return self.infills
        if self.use_archive is not None:
            with open(self.use_archive, "rb") as f:
                self.infills = pickle.load(f)
        else:
            # Get n_doe samples from sample space and evaluate
            self.infills = self.initialization.do(self.problem, self.n_doe, algorithm=self)
        return self.infills

    def _initialize_advance(self, infills=None, **kwargs):
        # Merge the new infills with the previously evaluated population
        self.infills = infills
        self._archive = Population.merge(self._archive, infills)
        with open(f"start_archive_{self.sbatch}.pkl", "wb") as f:
            pickle.dump(self._archive, f)

    def _advance(self, infills=None, **kwargs):
        if self.verbose: print(f'################### ADVANCING GENERATION {self.it} #########################')
        # Merge the current infills with the previously evaluated population
        self.infills = infills

        archive = copy.deepcopy(self._archive)
        self._archive = Population.merge(self._archive, infills)
        if self.verbose:
            self.plot_progress(infills, archive)
        self.it += 1

    def plot_progress(self, infills, archive):
        pass

    @staticmethod
    def pareto_line(front):
        pareto_sorted = front[np.argsort(front[:, 0])]
        # Create step-wise points
        x_step = []
        y_step = []

        for i in range(len(pareto_sorted) - 1):
            x_step.extend([pareto_sorted[i, 0], pareto_sorted[i + 1, 0]])  # Extend x with horizontal movement
            y_step.extend([pareto_sorted[i, 1], pareto_sorted[i, 1]])  # Extend y with constant value

        # Add last point to finish the front
        x_step.append(pareto_sorted[-1, 0])
        y_step.append(pareto_sorted[-1, 1])
        return np.array([x_step, y_step])

    def _calc_hv(self, F):
        # calculate hypervolume on the non-dominated set of F
        norm_F = self.normalize(F, objectives=self.problem.objectives.keys())

        front = NonDominatedSorting().do(norm_F, only_non_dominated_front=True)
        nd_F = norm_F[front, :]
        hv_metric = Hypervolume(ref_point=np.array([1.01, 1.01]),
                                norm_ref_point=False,
                                zero_to_one=True,
                                ideal=norm_F.min(axis=0),
                                nadir=norm_F.max(axis=0))

        hv = hv_metric.do(nd_F)
        return hv

    def normalize(self, F, objectives):
        F_norm = np.zeros_like(F)
        for o, obj in enumerate(objectives):
            if obj == 'Classification_Error':
                F_norm[:, o] = F[:, o] / 100
            elif obj == 'MACs' or obj == 'Parameters' or obj == 'mse':
                log = np.log10(F[:, o])
                F_norm[:, o] = (log - np.min(log)) / (np.max(log) - np.min(log))
            else:
                F_norm[:, o] = (F[:, o] - np.min(F[:, o])) / (np.max(F[:, o]) - np.min(F[:, o]))

        return F_norm

    def define_surrogate_algorithm(self):
        nsga_pop_size = self.sa_algorithm.get('population_size', 40)
        topx = int(nsga_pop_size * 0.75)

        topx_pop = RankAndCrowding().do(problem=self.problem, pop=self._archive, n_survive=topx)
        random_pop = self.sample_space(self.problem, nsga_pop_size - topx)
        infill_sample_space = Population.merge(topx_pop, random_pop)

        return NSGA2(pop_size=nsga_pop_size,
                     sampling=infill_sample_space,
                     eliminate_duplicates=self.sa_algorithm.get('dedup', False),
                     )

    def define_other_objectives(self):
        return {}

    def define_surrogate_problem(self, other_objective_functions):
        pass

    def _infill(self):
        if self.continue_id is not None:
            if self.continue_id[0] is not None:
                self.it = self.continue_id[1] + 1
                if self.verbose: print(
                    f'################### CONTINUE {self.continue_id[0]} - {self.continue_id[1]} #########################')
            self.continue_id = None

        if self.verbose: print('################### NEW INFILLS #########################')
        # Look for the next K number of candidates for low level evaluation
        # Get non-dominated architectures from archive:
        X = self._archive.get('X')
        F = self._archive.get('F')

        # Fit the surrogate model
        self._fit_surrogate(self._archive)

        front = NonDominatedSorting().do(F, only_non_dominated_front=True)

        algorithm = self.define_surrogate_algorithm()

        other_objective_functions = self.define_other_objectives()

        surrogate_problem = self.define_surrogate_problem(other_objective_functions)

        res = minimize(surrogate_problem,
                       algorithm,
                       ('n_gen', self.n_gen_candidates),
                       seed=self.seed,
                       save_history=True,
                       verbose=True)

        with open(f"{self.save_dir}/surrogate_result_{self.sbatch}.pkl", "wb") as f:
            pickle.dump(res, f)

        # check for duplicates
        norm_cand = self.normalize(res.pop.get("F"), self.problem.objectives.keys())
        norm_F = self.normalize(self._archive.get("F"), self.problem.objectives.keys())

        archive_genes = self.decode(self._archive.get('X'), real=False)
        res_genes = self.decode(res.pop.get('X'), real=False)
        combined = np.vstack((res_genes, archive_genes))
        unique_rows, idx = np.unique(combined, axis=0, return_index=True)
        not_duplicate = idx[idx < len(res_genes)]
        if self.verbose: print(f'Nr of non duplicates: {len(not_duplicate)}')

        # form a subset selection problem to short list K from pop_size
        indices = subset_selection(norm_cand[not_duplicate, 0], norm_F[front, 0], self.n_infill)
        if indices is None:  # Too many duplicates, fill with random samples
            infill_1 = res.pop[not_duplicate]
            infill_2 = self.sample_space(self.problem, self.n_infill - len(infill_1))
            infill = Population.merge(infill_1, infill_2)
        elif len(indices) < self.n_infill:
            infill_1 = res.pop[not_duplicate][indices]
            infill_2 = self.sample_space(self.problem, self.n_infill - len(infill_1))
            infill = Population.merge(infill_1, infill_2)
        else:
            infill = res.pop[not_duplicate][indices]

        self.predictions[self.it] = self.surrogate.predict(self.decode(infill.get('X'))).squeeze()

        # Make sure the candidates are evaluated with real fitness function
        candidates = Population().new("X", infill.get('X'))
        return candidates

    def _fit_surrogate(self, datapoints):
        X = datapoints.get('X')
        F = datapoints.get('F')

        # Remove significant outliers:
        F_prune = F[F[:, 0] > 0]
        X_prune = X[F[:, 0] > 0]

        assert len(X) > len(X[0]), "# of training samples have to be > # of dimensions"

        self.surrogate.fit(self.decode(X_prune), F_prune[:, 0])

    @staticmethod
    def decode(X, method='shifted', real=True):
        X = deepcopy(X)
        X_t = np.zeros(shape=(len(X), X[0, 0].normal, X[0, 0].n_var_size), dtype=float)
        X_tt = np.zeros(shape=(len(X), X[0, 0].normal * X[0, 0].n_var_size), dtype=float)
        Xi_t = np.zeros(shape=(len(X), X[0, 0].normal, X[0, 0].n_var_size), dtype=int)
        Xi_tt = np.zeros(shape=(len(X), X[0, 0].normal * X[0, 0].n_var_size), dtype=int)
        for i in range(len(X)):
            for n in range(X[0, 0].normal):
                X[i, 0].individual[n].toReal()
                X[i, 0].individual[n].realgene[~X[i, 0].individual[n].is_active] = np.NaN
                X_t[i, n] = X[i, 0].individual[n].realgene.flatten()
                for g, gene in enumerate(X[i, 0].individual[n].gene):
                    gen_length = len(X[i, 0].individual[n].gene[g]) + len(X[i, 0].individual[n].weight[g])
                    Xi_t[i, n, g * gen_length:(g + 1) * gen_length] = np.hstack(
                        [X[i, 0].individual[n].gene[g], X[i, 0].individual[n].weight[g]])
            if method == 'shifted':
                temp = X_t[i].flatten()
                tempi = Xi_t[i].flatten()
                nan_mask = np.isnan(temp)
                non_nans = temp[~nan_mask]
                X_tt[i][:len(non_nans)] = non_nans
                X_tt[i][len(non_nans):] = np.NaN
                Xi_tt[i][:len(non_nans)] = tempi[~nan_mask]

            else:
                X_tt[i] = X_t[i].flatten()
                Xi_tt[i] = Xi_t[i].flatten()

        if real:
            return X_tt
        else:
            return Xi_tt


class SurrogateProblem(Problem):
    def __init__(self,
                 search_space,
                 predictor,
                 objectives,
                 nvar_real):
        super().__init__(n_var=1, n_obj=len(objectives) + 1, n_constr=0, requires_kwargs=True)

        self.ss = search_space
        self.predictor = predictor
        self.objectives = objectives

        # Bounds for real-valued variables
        self.xl = np.zeros(nvar_real)
        self.xu = 0.999999 * np.ones(nvar_real)

    def _evaluate(self, x, out, *args, **kwargs):
        pass

def get_correlation(prediction, target):
    import scipy.stats as stats

    rmse = np.sqrt(((prediction - target) ** 2).mean())
    rho, _ = stats.spearmanr(prediction, target)
    tau, _ = stats.kendalltau(prediction, target)

    return rmse, rho, tau


class MyNormalization(ZeroToOneNormalization):

    def forward(self, X):
        return super().forward(X) * 200 - 100

    def backward(self, X):
        return super().backward((X + 100) / 200)
