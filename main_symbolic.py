import argparse
import os
import pickle
import random

import numpy as np
import sympy as sp
import torch
from pymoo.optimize import minimize
from pytorch_lightning.loggers import TensorBoardLogger

from strategy.surrogate.applications.symbolic import Symbolic_SAMOS


def main(args):
    # Set seed
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Optimizes runtime
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Logger
    if args.sbatch is not None:
        logger = {'save_dir': 'lightning_logs', 'name': 'CGP_SYMBOL', 'version': f'version_{args.sbatch}'}
        experiment_logger = TensorBoardLogger(**logger)
        plot_path = os.path.join(logger['save_dir'], logger['name'], logger['version'], 'plots/')
    else:
        logger = {'save_dir': 'lightning_logs', 'name': 'CGP_SYMBOL'}
        experiment_logger = TensorBoardLogger(**logger)
        logger['version'] = f'version_{experiment_logger.version}'
        plot_path = os.path.join(logger['save_dir'], logger['name'], logger['version'], 'plots/')

    print('plot path: ', plot_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    logger['plot_path'] = plot_path

    if args.id is not None:
        archive_it = int(args.id[:-4].split('_')[-1])
        continue_id = [args.id, archive_it]
    else:
        continue_id = [None, 0]

    #################################################
    ##################### Model #####################
    #################################################

    # Build modelspace
    from search_space.cgpnasv2.cgp import CGP
    search_space = CGP(functions='SYMBOLIC',
                       blocks=[{'rows': 10, 'cols': 4, 'range': [-10, 10]}])

    # Define evaluation method
    from evaluator.symbolic import MO_evaluation

    # Data gen
    x = sp.Symbol('x')
    x_data = np.linspace(0, 1, 10)
    # reference_function = 1.25 * sp.sin(x) + 6.3 * x ** 2
    reference_function = x ** 2 + 2 * x + 5
    y_data = np.array([reference_function.subs(x, val) for val in x_data], dtype=np.float64)

    if args.random:
        surrogate_search = Symbolic_SAMOS(sample_space=search_space.sample_space,
                                          problem_type='symbolic',
                                          n_doe=1000,
                                          n_infill=5,
                                          n_gen_candidates=1,
                                          n_gen_surrogate=2,
                                          n_var=search_space.get_problem_size(),
                                          logger_params=logger,
                                          sbatch=f'random_search_s{seed}_{args.sbatch if args.sbatch is not None else logger["version"]}',
                                          sa_algorithm={'name': 'NSGA2',
                                                        'population_size': 40,
                                                        'mutation_prob': 0.3,
                                                        'crossover_prob': 0.9,
                                                        'dedup': False,
                                                        'seed': seed}
                                          )
    else:

        surrogate_search = Symbolic_SAMOS(sample_space=search_space.sample_space,
                                          problem_type='symbolic',
                                          n_doe=100,
                                          n_infill=20,
                                          n_gen_candidates=20,
                                          n_gen_surrogate=25,
                                          n_var=search_space.get_problem_size(),
                                          logger_params=logger,
                                          sbatch=args.sbatch,
                                          # use_archive='start_archive_347715.pkl',
                                          continue_id=continue_id,
                                          sa_algorithm={'name': 'NSGA2',
                                                        'population_size': 100,
                                                        'mutation_prob': 0.3,
                                                        'crossover_prob': 0.9,
                                                        'dedup': False,
                                                        'seed': seed}
                                          )

    evaluator = MO_evaluation(objectives={'mse': '', 'complexity': ''},
                              nvar_real=search_space.get_problem_size(),
                              x_data=x_data,
                              y_data=y_data
                              )

    res = minimize(evaluator,
                   surrogate_search,
                   termination=('n_gen', surrogate_search.n_gen_surrogate),
                   save_history=True,
                   verbose=True
                   )

    # Save results to a file
    output_filename = f"Optimized_SAMOS_CGP_symbolic_Results_{args.sbatch}.pkl"
    with open(output_filename, mode='wb') as file:
        pickle.dump(res, file)

    # obtain the result objective from the algorithm
    print(res.X)
    print(res.F)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--config",
                        dest="config_file",
                        help="Select Training Config File")
    parser.add_argument("-i",
                        "--checkpoint_id",
                        dest="id",
                        help="checkpoint_folder",
                        default=None)
    parser.add_argument("-n",
                        "--node",
                        dest="node",
                        help="node",
                        default=None)

    parser.add_argument("-b",
                        "--sbatch",
                        dest="sbatch",
                        help="sbatch",
                        default=None)

    parser.add_argument("-r",
                        "--random",
                        dest="random",
                        help="random",
                        action="store_true")

    arguments = parser.parse_args()
    print(f'arguments: {arguments}')

    main(arguments)
