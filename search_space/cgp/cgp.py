### Based on: https://github.com/Cosijopiii/CGPNAS
from typing import List, Union
from search_space.cgp.configuration import ConfigureCGP
from search_space.cgp.sampling import SamplingCGP
import search_space.cell_options as ops

# from search_space.cgpnasv2.CartesianDefinitionW import CartesianGPConfigurationW
# from search_space.cgpnasv2.CartesianCellSamplingW import CartesianCellGeneticProgrammingW
#
# from search_space.symbolic.CGPConfiguration import CartesianGPConfigurationB
# from search_space.symbolic.CGPSample import CartesianCellGeneticProgrammingB



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

UNARY_FUNCTIONS = [
    'sin',
    'cos',
    'exp',
    'log',
    'constant',
]

BINARY_FUNCTIONS = [
    'sqrt',
    'square',
    'add',
    'mul',
    'sub',
    'div',
]


class CGP:
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
            self.sampler = CartesianCellGeneticProgrammingW
        elif self.directive == 'CGPNASV1':
            self.given_functions = CGPNASV1_FUNCTIONS
            self.sampler = CartesianCellGeneticProgrammingW
        elif self.directive == 'SYMBOLIC':
            self.given_functions = UNARY_FUNCTIONS + BINARY_FUNCTIONS
            self.sampler = CartesianCellGeneticProgrammingB

        if self.directive is 'SYMBOLIC':
            self.functions = self.given_functions
            self.arity = [1 if function in UNARY_FUNCTIONS else 2 for function in self.functions]

        elif self.directive is not None:
            if not set(self.given_functions).issubset(ops.__all__):
                missing_items = [item for item in self.given_functions if item not in ops.__all__]
                raise ValueError(f"The {missing_items} are not specified in the cell options.")

            self.functions = [function for function in dir(ops) if function in self.given_functions]

            self.arity = [1 if (function != 'Sum' and function != 'Concatenate') else 2 for function in self.functions]

    def configure_cgp_layer(self, rows=10, cols=4, channels=(32, 64)):
        if self.directive == 'SYMBOLIC':
            return CartesianGPConfigurationB(
                rows=rows,
                cols=cols,
                level_back=1,
                min_active_num=1,
                max_active_num=0,
                funcs=self.functions,
                funAry=self.arity,
                constant_range=channels
            )
        return CartesianGPConfigurationW(
            rows=rows,
            cols=cols,
            level_back=1,
            min_active_num=1,
            max_active_num=0,
            funcs=self.functions,
            funAry=self.arity,
            channels=channels
        )

    def configure_blocks(self, blocks):
        self.blocks = []
        for block in blocks:
            try:
                self.blocks.append(self.configure_cgp_layer(block['rows'], block['cols'], block['channels']))
            except:
                self.blocks.append(self.configure_cgp_layer(block['rows'], block['cols'], block['range']))
        return self.blocks

    def get_problem_size(self):
        if self.directive == 'SYMBOLIC':
            n_weight = 1
        else:
            n_weight = 2
        return (self.blocks[0].node_num + self.blocks[0].out_num) * (self.blocks[0].max_in_num + 1 + n_weight)

    def configure_sample_space(self):
        if self.directive == 'SYMBOLIC':
            return self.sampler(self.blocks, len(self.blocks))

        return self.sampler(self.blocks, self.model_pool, len(self.blocks), len(self.blocks) - 1)
