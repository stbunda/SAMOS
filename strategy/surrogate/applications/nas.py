import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from search_space.cgpnasv2.CGPDecoder import CGPDecoder as CGPDecoder_original
from search_space.cgpnasv2.CGPDecoder_new import CGPDecoder
from strategy.cgpnas.CrossoverCellCgpW import CrossoverCellCgpW
from strategy.cgpnas.MutationCellCgpW import MutationCellCgpW
from strategy.surrogate.samos import SAMOS, get_correlation, SurrogateProblem


class NAS_SAMOS(SAMOS):
    def __init__(self,
                 decoder_style='original',
                 **kwargs):

        super().__init__(**kwargs)
        self.decoder_style = decoder_style

    def plot_progress(self, infills, archive):
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

    def define_surrogate_algorithm(self):
        nsga_pop_size = self.sa_algorithm.get('population_size', 40)
        topx = int(nsga_pop_size * 0.75)

        topx_pop = RankAndCrowding().do(problem=self.problem, pop=self._archive, n_survive=topx)
        random_pop = self.sample_space(self.problem, nsga_pop_size - topx)
        infill_sample_space = Population.merge(topx_pop, random_pop)

        crossover = self.sa_algorithm.get('crossover_prob', 0.9)
        mutation = self.sa_algorithm.get('mutation', 0.3)
        crossover_eta = self.sa_algorithm.get('crossover_eta', 20)
        mutation_eta = self.sa_algorithm.get('mutation_eta', 15)

        return NSGA2(
            pop_size=nsga_pop_size,
            sampling=infill_sample_space,
            crossover=CrossoverCellCgpW(prob=crossover, eta=crossover_eta),
            mutation=MutationCellCgpW(prob=mutation, eta=mutation_eta),
            eliminate_duplicates=self.sa_algorithm.get('dedup', False),
        )

    def define_other_objectives(self):
        return {
            'MAC': self.problem.count_macs if hasattr(self.problem, 'count_macs') else None,
            'Parameters': self.problem.count_parameters if hasattr(self.problem, 'count_parameters') else None,
        }

    def define_surrogate_problem(self, other_objective_functions):
        return NasSurrogateProblem(search_space=self.sample_space,
                                   predictor=self.surrogate,
                                   objectives={key: other_objective_functions.get(key, "") for key in
                                               self.problem.objectives.keys() & other_objective_functions},
                                   input_tensor=self.problem.datamodule.input_tensor,
                                   n_classes=len(self.problem.datamodule.classes),
                                   decoder=self.decode,
                                   decoder_style=self.decoder_style,
                                   nvar_real=self.n_var,
                                   )


class NasSurrogateProblem(SurrogateProblem):
    """ The optimization problem for finding the next N candidate architectures """

    def __init__(self,
                 input_tensor,
                 n_classes,
                 decoder,
                 decoder_style,
                 **kwargs):
        super().__init__(**kwargs)

        self.input_tensor = input_tensor
        self.n_classes = n_classes
        self.decoder = decoder
        self.decoder_style = decoder_style

    def _evaluate(self, x, out, *args, **kwargs):
        metrics = np.zeros((len(x), self.n_obj))
        metrics[:, 0] = self.predictor.predict(self.decoder(x)).squeeze()

        for m, model_instance in enumerate(x):
            if self.decoder_style == 'original':
                model = CGPDecoder_original(model_instance[0].active_net_list(), self.input_tensor, self.n_classes)
            else:
                model = CGPDecoder(model_instance[0].active_net_list(), self.input_tensor, self.n_classes)
            for o, objective in enumerate(list(self.objectives.keys())):
                metrics[m, o + 1] = self.objectives[objective](model)
        out['F'] = metrics
