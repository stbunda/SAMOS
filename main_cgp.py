import argparse
import os
import pickle
import random

import numpy as np
import torch
from pymoo.optimize import minimize
from pytorch_lightning.loggers import TensorBoardLogger


def main(args):
    # Set seed
    seed = 89 if args.seed is None else int(args.seed)
    print('seed: ', seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Optimizes runtime
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Logger
    if args.sbatch is not None:
        logger = {'save_dir': 'lightning_logs', 'name': 'CGP', 'version': f'version_{args.sbatch}'}
    else:
        logger = {'save_dir': 'lightning_logs', 'name': 'CGP'}
    experiment_logger = TensorBoardLogger(**logger)
    logger['version'] = str(experiment_logger.version)

    plot_path = os.path.join(logger['save_dir'], logger['name'], logger['version'], 'plots/')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    logger['plot_path'] = plot_path
    if args.sbatch is None:
        args.sbatch = logger['version']

    if args.id is not None:
        archive_it = int(args.id[:-4].split('_')[-1])
        continue_id = [args.id, archive_it]
    else:
        continue_id = [None, 0]

    #################################################
    ##################### Model #####################
    #################################################

    # Build modelspace
    from search_space.cgpnas.cgp import CGP
    search_space = CGP(functions='CGPNASV2',
                       blocks=[{'rows': 10, 'cols': 4, 'channels': [32, 64]},
                               {'rows': 10, 'cols': 4, 'channels': [64, 128]},
                               {'rows': 10, 'cols': 4, 'channels': [128, 256]},
                               ],
                       model_pool=[1, 1])

    # Define evaluation method
    from evaluator.nas_classification import MO_evaluation
    from evaluator.dataset.pytorch_dataset import CIFAR10

    dataset = CIFAR10(phases=['fit', 'test'],
                      data_dir='data',
                      train_val_split=0.2,
                      generator_seed=seed,
                      # subset=100
                      )

    dataset.setup('fit')
    dataset.train_dataloader(batch_size=128,
                             shuffle=True,
                             drop_last=True,
                             num_workers=8,
                             pin_memory=True)
    dataset.test_dataloader(batch_size=128,
                            num_workers=8,
                            pin_memory=True)

    from strategy.cgpnas.cgp import CGPNASV2
    cgp_algorithm = CGPNASV2(population_size=20,
                             generations=30,
                             seed=seed,
                             )
    strategy = cgp_algorithm.define_algorithm(search_space.sample_space)

    evaluator = MO_evaluation(max_epochs=10,
                              datamodule=dataset,
                              num_classes=len(dataset.classes),
                              devices=-1,
                              accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                              logger_params=logger,
                              batch_nr=args.sbatch,
                              objectives={'Classification_Error': {'learning_rate': 0.025,
                                                                   'weight_decay': 3.0e-4},
                                          'MAC': {'scale': 1.0e+6}
                                          },
                              callbacks={'EarlyStopping': {'monitor': 'val_accuracy',
                                                           'min_delta': 0.0,
                                                           'patience': 3}},
                              nvar_real=search_space.get_problem_size(),
                              layer_options=search_space.given_functions,
                              generations=cgp_algorithm.generations,
                              )

    res = minimize(evaluator,
                   strategy,
                   termination=('n_gen', cgp_algorithm.generations - continue_id[1]),
                   save_history=True,
                   verbose=True
                   )

    # Save results to a file
    output_filename = f"Optimized_CGP_Results_{args.sbatch}.pkl"
    with open(output_filename, mode='wb') as file:
        pickle.dump(res, file)

    # obtain the result objective from the algorithm
    print(res.X)
    print(res.F)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--checkpoint_id",
                        dest="id",
                        help="checkpoint_folder",
                        default=None)
    parser.add_argument("-b",
                        "--sbatch",
                        dest="sbatch",
                        help="sbatch",
                        default=None)
    parser.add_argument("-d",
                        "--decoder",
                        dest="decoder",
                        help="decoder",
                        default='old')
    parser.add_argument("-r",
                        "--random",
                        dest="random",
                        help="random",
                        action="store_true")
    parser.add_argument("-s",
                        "--seed",
                        dest="seed",
                        help="seed",
                        default=None)
    parser.add_argument("-a",
                        "--archive",
                        dest="archive",
                        help="archive",
                        default=None)

    arguments = parser.parse_args()

    main(arguments)
