import copy
import os
import pickle
from copy import deepcopy

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
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

from search_space.cgpnas.CGPDecoder import CGPDecoder as CGPDecoder_original
from search_space.cgpnas.CGPDecoder_new import CGPDecoder
from search_space.symbolic.symbolic_regression import CGPDecoder as CGPDecoder_SR
from models.carts import CART
from subset_selection import subset_selection
from ..cgpnas.CrossoverCellCgpW import CrossoverCellCgpW
from ..cgpnas.MutationCellCgpW import MutationCellCgpW
from ..symbolic.CrossoverCellCgpB import CrossoverCellCgpB
from ..symbolic.MutationCellCgpB import MutationCellCgpB

matplotlib.use('Agg')


class SAMOS(Algorithm):
    def __init__(self,
                 sample_space: Sampling,  # Define search space
                 problem: Problem = None,  # Define search problem
                 problem_type: str = 'NAS',
                 n_doe: int = 100,  # Nr of initial architectures to fit surrogate
                 n_infill: int = 8,  # Nr of architectures to train
                 n_gen_candidates: int = 20,  # Nr of generations of NSGA-II loop
                 n_gen_surrogate: int = 30,  # Nr of generations of the surrogate model
                 n_var: int = None,  # Nr of variables
                 logger_params=None,
                 sbatch='',
                 decoder_style='old',
                 use_archive=None,
                 continue_id=None,
                 nsga_params=None,  # Genetic algorithm parameters
                 **kwargs):

        super().__init__(eliminate_duplicates=False, **kwargs)
        self.sample_space = sample_space
        self.problem = problem
        self.problem_type = problem_type
        self.logger_params = logger_params
        self.nsga_params = nsga_params
        self.sbatch = sbatch
        self.decoder_style = decoder_style
        self.use_archive = use_archive
        self.continue_id = continue_id

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
        print(f'################### ADVANCING GENERATION {self.it} #########################')
        # Merge the current infills with the previously evaluated population
        # Fc = self.problem.evaluate(infills.get('X'))
        self.infills = infills

        archive = copy.deepcopy(self._archive)
        self._archive = Population.merge(self._archive, infills)

        if self.problem.problem_type == 'nas':
            self.plot_progress_nas(infills, archive)
        else:
            self.plot_progess_symbol(infills, archive)

        self.it += 1

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

    def plot_progress_nas(self, infills, archive):
        F = self._archive.get('F')
        Fa = archive.get('F')
        Fc = infills.get('F')

        # Error predictions:
        a_error_pred = self.surrogate.predict(self.decode(archive.get('X')))
        c_error_pred = self.surrogate.predict(self.decode(infills.get('X')))

        # Actual error
        a_error = Fa[:, 0]
        c_error = Fc[:, 0]

        # check for accuracy predictor's performance
        rmse, rho, tau = get_correlation(
            np.vstack((a_error_pred, c_error_pred)),
            np.vstack((a_error.reshape(-1, 1), c_error.reshape(-1, 1)))
        )
        print(f"fitting {self.surrogate}: RMSE = {rmse:.4f}, Spearmans Rho = {rho:.4f}, Kendalls Tau = {tau:.4f}")

        path = os.path.join(self.save_dir, str(self.it))

        if not os.path.exists(path):
            os.makedirs(path)

        with open(f"{path}/archive_{self.sbatch}_{self.it}.pkl", "wb") as f:
            pickle.dump(self._archive, f)
        with open(f"{path}/surrogate_{self.sbatch}_{self.it}.pkl", "wb") as f:
            pickle.dump(self.surrogate, f)
        with open(f"{path}/infills_{self.sbatch}_{self.it}.pkl", "wb") as f:
            pickle.dump(infills, f)

        # calculate hypervolume
        hv = self._calc_hv(F)
        pF = F[NonDominatedSorting().do(F, only_non_dominated_front=True)]
        pFc = Fc[NonDominatedSorting().do(Fc, only_non_dominated_front=True)]
        pFa = Fa[NonDominatedSorting().do(Fa, only_non_dominated_front=True)]

        # print iteration-wise statistics
        print("Iter {}: hv = {:.2f}".format(self.it, hv))
        F_line = self.pareto_line(pF)
        Fa_line = self.pareto_line(pFa)

        for i in range(2):
            plt.figure(figsize=(18, 10))
            plt.scatter(Fa[:, 0], Fa[:, 1] * 1e6, alpha=0.3, label='Archive', color='green')
            plt.plot(Fa_line[0, :], Fa_line[1, :] * 1e6, label='NDS Archive', marker='*', color='green')
            plt.scatter(pFa[:, 0], pFa[:, 1] * 1e6, label='NDS Archive', marker='*', color='green')

            plt.scatter(self.predictions[self.it], Fc[:, 1] * 1e6, label=f'Predictions Gen {self.it}', marker='v',
                        color='orange')

            plt.scatter(Fc[:, 0], Fc[:, 1] * 1e6, alpha=0.7, label=f'Evaluated Gen {self.it}', color='blue')
            plt.plot(F_line[0, :], F_line[1, :] * 1e6, label=f'NDS Evaluated Gen {self.it}', marker='*', color='blue')
            plt.scatter(pFc[:, 0], pFc[:, 1] * 1e6, label=f'NDS Evaluated Gen {self.it}', marker='*', color='blue')

            plt.scatter(pF[:, 0], pF[:, 1] * 1e6, label=f'NDS total population', marker='*')

            plt.legend(loc='upper right')
            plt.xlabel('% Classification Error')

            if i == 0:
                plt.yscale('log')
                plt.ylabel('MAdds')
                plt.title("Objective Space")
                plt.savefig(os.path.join(self.logger_params['plot_path'], f"log_objective_space_s{self.it}.png"))
                plt.close()
            else:
                plt.ylabel('MAdds')
                plt.title("Objective Space")
                plt.savefig(os.path.join(self.logger_params['plot_path'], f"objective_space_s{self.it}.png"))
                plt.close()

        plt.figure(figsize=(18, 10))
        plt.scatter(a_error, a_error_pred, label='Archive')
        plt.scatter(c_error, c_error_pred, label=f'Gen {self.it}')
        min_val = np.min([np.min(a_error), np.min(a_error_pred)])
        max_val = np.max([np.max(a_error), np.max(a_error_pred)])

        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x (Ideal)")
        plt.xlabel('Classification Error')
        plt.ylabel('Predicted Error')
        plt.legend()
        plt.title("Surrogate Error Prediction")
        plt.savefig(os.path.join(self.logger_params['plot_path'], f"error_prediction_s{self.it}.png"))
        plt.close()

    def plot_progess_symbol(self, infills, archive):
        F = self._archive.get('F')
        F = F[F[:, 0] > 0]

        Fa = archive.get('F')
        Fa_error_idx = Fa[:, 0] > 0
        Fa = Fa[Fa_error_idx]

        Fc = infills.get('F')
        Fc_error_idx = Fc[:, 0] > 0
        Fc = Fc[Fc_error_idx]

        # Error predictions:
        a_error_pred = self.surrogate.predict(self.decode(archive.get('X')))
        a_error_pred = a_error_pred[Fa_error_idx]

        c_error_pred = self.surrogate.predict(self.decode(infills.get('X')))
        c_error_pred = c_error_pred[Fc_error_idx]

        # Actual error
        a_error = Fa[:, 0]
        c_error = Fc[:, 0]

        # check for accuracy predictor's performance
        rmse, rho, tau = get_correlation(
            np.vstack((a_error_pred, c_error_pred)),
            np.vstack((a_error.reshape(-1, 1), c_error.reshape(-1, 1)))
        )
        print(f"fitting {self.surrogate}: RMSE = {rmse:.4f}, Spearmans Rho = {rho:.4f}, Kendalls Tau = {tau:.4f}")

        path = os.path.join(self.save_dir, str(self.it))

        if not os.path.exists(path):
            os.makedirs(path)

        with open(f"{path}/archive_{self.sbatch}_{self.it}.pkl", "wb") as f:
            pickle.dump(self._archive, f)
        with open(f"{path}/surrogate_{self.sbatch}_{self.it}.pkl", "wb") as f:
            pickle.dump(self.surrogate, f)
        with open(f"{path}/infills_{self.sbatch}_{self.it}.pkl", "wb") as f:
            pickle.dump(infills, f)

        # calculate hypervolume
        hv = self._calc_hv(F)
        pF = F[NonDominatedSorting().do(F, only_non_dominated_front=True)]
        pFc = Fc[NonDominatedSorting().do(Fc, only_non_dominated_front=True)]
        pFa = Fa[NonDominatedSorting().do(Fa, only_non_dominated_front=True)]

        # print iteration-wise statistics
        print("Iter {}: hv = {:.2f}".format(self.it, hv))
        F_line = self.pareto_line(pF)
        Fa_line = self.pareto_line(pFa)

        for i in range(2):
            plt.figure(figsize=(18, 10))
            plt.scatter(Fa[:, 0], Fa[:, 1], alpha=0.3, label='Archive', color='green')
            plt.plot(Fa_line[0, :], Fa_line[1, :], label='NDS Archive', marker='*', color='green')
            plt.scatter(pFa[:, 0], pFa[:, 1], label='NDS Archive', marker='*', color='green')

            plt.scatter(self.predictions[self.it][Fc_error_idx], Fc[:, 1], label=f'Predictions Gen {self.it}',
                        marker='v',
                        color='orange')

            plt.scatter(Fc[:, 0], Fc[:, 1], alpha=0.7, label=f'Evaluated Gen {self.it}', color='blue')
            plt.plot(F_line[0, :], F_line[1, :], label=f'NDS Evaluated Gen {self.it}', marker='*', color='blue')
            plt.scatter(pFc[:, 0], pFc[:, 1], label=f'NDS Evaluated Gen {self.it}', marker='*', color='blue')

            plt.scatter(pF[:, 0], pF[:, 1], label=f'NDS total population', marker='*')

            plt.legend(loc='upper right')
            plt.xlabel('MSE')
            # plt.xlim([0, 10000])
            if i == 0:
                plt.xscale('log')

                plt.ylabel('Complexity')
                plt.title("Objective Space")
                plt.savefig(os.path.join(self.logger_params['plot_path'], f"log_objective_space_s{self.it}.png"))
                plt.close()
            else:
                plt.ylabel('Complexity')
                plt.title("Objective Space")
                plt.savefig(os.path.join(self.logger_params['plot_path'], f"objective_space_s{self.it}.png"))
                plt.close()

        plt.figure(figsize=(18, 10))
        plt.scatter(a_error, a_error_pred, label='Archive')
        plt.scatter(c_error, c_error_pred, label=f'Gen {self.it}')
        min_val = np.min([np.min(a_error), np.min(a_error_pred)])
        max_val = np.max([np.max(a_error), np.max(a_error_pred)])

        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x (Ideal)")
        plt.xlabel('Classification Error')
        plt.xscale('log')
        plt.ylabel('Predicted Error')
        plt.yscale('log')
        plt.legend()
        plt.title("Surrogate Error Prediction")
        plt.savefig(os.path.join(self.logger_params['plot_path'], f"error_prediction_s{self.it}.png"))
        plt.close()

    def _calc_hv(self, F):
        # calculate hypervolume on the non-dominated set of F
        norm_F = self.normalize(F, objectives=self.problem.objectives.keys())

        front = NonDominatedSorting().do(norm_F, only_non_dominated_front=True)
        nd_F = norm_F[front, :]
        # ref_point = 1.01 * np.max(norm_F, axis=0)

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

    def _infill(self):
        if self.continue_id is not None:
            if self.continue_id[0] is not None:
                self.it = self.continue_id[1] + 1
                print(
                    f'################### CONTINUE {self.continue_id[0]} - {self.continue_id[1]} #########################')
            self.continue_id = None

        print('################### NEW INFILLS #########################')
        # Look for the next K number of candidates for low level evaluation
        # Get non-dominated architectures from archive:
        X = self._archive.get('X')
        F = self._archive.get('F')

        # Fit the surrogate model
        self._fit_surrogate(self._archive)

        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        # front_X = X[front]

        nsga_pop_size = self.nsga_params.get('population_size', 40)
        topx = int(nsga_pop_size * 0.75)

        topx_pop = RankAndCrowding().do(problem=self.problem, pop=self._archive, n_survive=topx)
        random_pop = self.sample_space(self.problem, nsga_pop_size - topx)
        infill_sample_space = Population.merge(topx_pop, random_pop)

        crossover = self.nsga_params.get('crossover_prob', 0.9)
        mutation = self.nsga_params.get('mutation', 0.3)
        crossover_eta = self.nsga_params.get('crossover_eta', 20)
        mutation_eta = self.nsga_params.get('mutation_eta', 15)

        if self.problem.problem_type == 'nas':
            algorithm = NSGA2(
                pop_size=nsga_pop_size,
                sampling=infill_sample_space,
                crossover=CrossoverCellCgpW(prob=crossover, eta=crossover_eta),
                mutation=MutationCellCgpW(prob=mutation, eta=mutation_eta),
                eliminate_duplicates=self.nsga_params.get('dedup', False),
            )
        else:
            algorithm = NSGA2(
                pop_size=nsga_pop_size,
                sampling=infill_sample_space,
                crossover=CrossoverCellCgpB(prob=crossover, eta=crossover_eta),
                mutation=MutationCellCgpB(prob=mutation, eta=mutation_eta),
                eliminate_duplicates=self.nsga_params.get('dedup', False),
            )

        other_objective_functions = {
            'MAC': self.problem.count_macs if hasattr(self.problem, 'count_macs') else None,
            'Parameters': self.problem.count_parameters if hasattr(self.problem, 'count_parameters') else None,
            'complexity': self.problem.calc_complexity if hasattr(self.problem, 'calc_complexity') else None,

        }
        if self.problem.problem_type == 'nas':
            problem = SurrogateProblem(search_space=self.sample_space,
                                       predictor=self.surrogate,
                                       objectives={key: other_objective_functions.get(key, "") for key in
                                                   self.problem.objectives.keys() & other_objective_functions},
                                       input_tensor=self.problem.datamodule.input_tensor,
                                       n_classes=len(self.problem.datamodule.classes),
                                       decoder=self.decode,
                                       decoder_style=self.decoder_style,
                                       nvar_real=self.n_var,
                                       )
        else:
            problem = SurrogateProblem2(search_space=self.sample_space,
                                        predictor=self.surrogate,
                                        objectives={key: other_objective_functions.get(key, "") for key in
                                                    self.problem.objectives.keys() & other_objective_functions},
                                        x_data=self.problem.x_data,
                                        y_data=self.problem.y_data,
                                        decoder=self.decode,
                                        nvar_real=self.n_var,
                                        )

        res = minimize(problem,
                       algorithm,
                       ('n_gen', self.n_gen_candidates),
                       seed=self.seed,
                       save_history=True,
                       verbose=True)

        # path = os.path.join(self.logger_params['save_dir'],
        #                     self.logger_params['name'],
        #                     f"version_{self.logger_params['version']}",
        #                     'archive')

        # if not os.path.exists(path):
        #     os.makedirs(path)
        #
        with open(f"{self.save_dir}/surrogate_result_{self.sbatch}.pkl", "wb") as f:
            pickle.dump(res, f)

        # check for duplicates
        # not_duplicates = np.logical_not(x_new in X for x_new in res.pop.get("X"))
        # todo: remove duplicates

        norm_cand = self.normalize(res.pop.get("F"), self.problem.objectives.keys())
        norm_F = self.normalize(self._archive.get("F"), self.problem.objectives.keys())

        archive_genes = self.decode(self._archive.get('X'), real=False)
        res_genes = self.decode(res.pop.get('X'), real=False)
        combined = np.vstack((res_genes, archive_genes))
        unique_rows, idx = np.unique(combined, axis=0, return_index=True)
        not_duplicate = idx[idx < len(res_genes)]
        print(f'Nr of non duplicates: {len(not_duplicate)}')
        # print('Not removing them')

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
        # todo: Set non-active path to NaN
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
    """ The optimization problem for finding the next N candidate architectures """

    def __init__(self,
                 search_space,
                 predictor,
                 objectives,
                 input_tensor,
                 n_classes,
                 nvar_real,
                 decoder,
                 decoder_style):
        super().__init__(n_var=1, n_obj=len(objectives) + 1, n_constr=0, requires_kwargs=True)

        self.ss = search_space
        self.predictor = predictor
        self.input_tensor = input_tensor
        self.n_classes = n_classes
        self.decoder = decoder
        self.decoder_style = decoder_style
        self.objectives = objectives

        # Bounds for real-valued variables
        self.xl = np.zeros(nvar_real)
        self.xu = 0.999999 * np.ones(nvar_real)

    def _evaluate(self, x, out, *args, **kwargs):
        metrics = np.zeros((len(x), self.n_obj))
        metrics[:, 0] = self.predictor.predict(self.decoder(x)).squeeze()

        for m, model_instance in enumerate(x):
            if self.decoder_style == 'old':
                model = CGPDecoder_original(model_instance[0].active_net_list(), self.input_tensor, self.n_classes)
            else:
                model = CGPDecoder(model_instance[0].active_net_list(), self.input_tensor, self.n_classes)
            for o, objective in enumerate(list(self.objectives.keys())):
                metrics[m, o + 1] = self.objectives[objective](model)
        out['F'] = metrics


class SurrogateProblem2(Problem):
    """ The optimization problem for finding the next N candidate architectures """

    def __init__(self,
                 search_space,
                 predictor,
                 objectives,
                 x_data,
                 y_data,
                 nvar_real,
                 decoder):
        super().__init__(n_var=1, n_obj=len(objectives) + 1, n_constr=0, requires_kwargs=True)

        self.ss = search_space
        self.predictor = predictor
        self.x_data = x_data
        self.y_data = y_data
        self.decoder = decoder
        self.objectives = objectives

        # Bounds for real-valued variables
        self.xl = np.zeros(nvar_real)
        self.xu = 0.999999 * np.ones(nvar_real)

    def _evaluate(self, x, out, *args, **kwargs):
        metrics = np.zeros((len(x), self.n_obj))
        metrics[:, 0] = self.predictor.predict(self.decoder(x)).squeeze()

        for m, model_instance in enumerate(x):
            model = CGPDecoder_SR(model_instance[0].active_net_list())
            for o, objective in enumerate(list(self.objectives.keys())):
                metrics[m, o + 1] = self.objectives[objective](model)
        out['F'] = np.array(metrics)


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
