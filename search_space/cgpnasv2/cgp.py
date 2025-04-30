### Based on: https://github.com/Cosijopiii/CGPNAS
from typing import List, Union

from search_space.cgp.configuration import ConfigureCGP
from search_space.cgp.sampling import SamplingCGP

import search_space.cell_options as ops

CGPNASV2_FUNCTIONS = [
    # Static operations
    'Identity', 'Sum', 'Concatenate',
    # Conv operations
    'ConvBlock', 'ResBlock', 'SepBlock', 'DilBlock', 'Bottleneck', 'Conv1x7_7x1',
    'MBConvBlock', 'FusedMBConvBlock'
]

CGPNASV1_FUNCTIONS = [
    # Static operations
    'Identity', 'Sum', 'Concatenate',
    # Conv operations
    'ConvBlock', 'ResBlock'
]

class CGPNAS:
    def __init__(self,
                 blocks: List,
                 functions: Union[List, str] = None,
                 model_pool: List = None):
        self.functions = []
        self.given_functions = functions
        self.directive = functions
        self.arity = []
        self.subset_functions()
        self.blocks = self.configure_blocks(blocks)
        self.model_pool = model_pool
        self.sample_space = self.configure_sample_space()

    def subset_functions(self):
        """
            Filters the list ARITY_FUNCTIONS based on a subset f of FUNCTION_LIST.

            Parameters:
                f (list): The subset of FUNCTION_LIST to filter.
            """
        if self.directive == 'CGPNASV2':
            self.given_functions = CGPNASV2_FUNCTIONS
            self.sampler = SamplingCGP
        elif self.directive == 'CGPNASV1':
            self.given_functions = CGPNASV1_FUNCTIONS
            self.sampler = SamplingCGP

        elif self.directive is not None:
            if not set(self.given_functions).issubset(ops.__all__):
                missing_items = [item for item in self.given_functions if item not in ops.__all__]
                raise ValueError(f"The {missing_items} are not specified in the cell options.")

            self.functions = [function for function in dir(ops) if function in self.given_functions]

            self.arity = [1 if (function != 'Sum' and function != 'Concatenate') else 2 for function in self.functions]

    def configure_cgp_layer(self, rows=10, cols=4, kernels=(3, 5), channels=(32, 64)):
        return ConfigureCGP(rows = rows,
                            cols = cols,
                            level_back = 1,
                            min_active_num = 1,
                            max_active_num = 0,
                            funcs = self.functions,
                            func_in_num = self.arity,
                            func_red = ['MaxPool'],
                            func_out = ['Output'],
                            weight_ops = {'kernels': kernels,
                                          'channels': channels}
                            )


    def configure_blocks(self, blocks):
        self.blocks = []
        for block in blocks:
            self.blocks.append(self.configure_cgp_layer(block['rows'], block['cols'], block['channels']))
        return self.blocks

    def get_problem_size(self):
        total_nodes = self.blocks[0].node_num + self.blocks[0].outputs  # nodes in grid + nr of output nodes
        node_size = 1 + self.blocks[0].input_max + self.blocks[0].nr_weights  # function + nr_connections + weights
        return total_nodes * node_size

    def configure_sample_space(self):
        return self.sampler(self.blocks, self.model_pool, len(self.blocks), len(self.blocks) - 1, weights=self.blocks[0].nr_weights)
