import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn import functional as F
import torch
from torchmetrics import Accuracy
from functools import partial
from .symbolic_regression_operators import unary_operators, binary_operators

import sympy as sp



def layer_functions(layer, outputs):
    return partial(layer, *outputs)


class CGPDecoder:
    def __init__(self, netCGP):
        super().__init__()

        self.netCGP = netCGP

        self.default_operators = unary_operators | binary_operators

        self.net = self.build_expr(len(self.netCGP) - 1)

        if self.net.has(sp.zoo):
            self.net = None


    def build_expr(self, node_id):
        if node_id == 0:  # First node is always 'x'
            return sp.Symbol('x')

        op, c1, c2, constant = self.netCGP[node_id]

        if op == 'constant':
            return sp.Float(constant)

        # Process first connection
        val1 = self.build_expr(c1)

        # Process second connection (if binary operation)
        if op in binary_operators:
            val2 = self.build_expr(c2)
        else:
            val2 = None



        # Apply operation
        return self.default_operators[op](val1) if val2 is None else self.default_operators[op](val1, val2)

if __name__ == "__main__":
    nodes = [
        ['input', 'x', 0, 0],
        ['sin', 0, 0, 0],  # Node 0: sin(x)
        ['square', 1, 0, 0],  # Node 1: sin(x)²
        ['exp', 0, 0, 0],  # Node 2: exp(x)
        ['mul', 2, 3, 0], # Node 3: sin(x)² * exp(x)
        ['constant', 0, 0, 5],  # Node 4: 5  <-- Final result
        ['mul', 4, 5, 0],  # Node 5: sin(x)² * exp(x) * 5 <-- Final result
    ]

    x = sp.Symbol('x')
    x_data = np.linspace(0, 5, 100)
    reference_function = sp.sin(x) ** 2 * sp.exp(x) * 5
    y_data = np.array([reference_function.subs(x, val) for val in x_data], dtype=np.float64)

    sym = CGPDecoder(nodes)

    y_hat = np.array([sym.net.subs(x, val) for val in x_data], dtype=np.float64)

    plt.figure()
    plt.plot(x_data, y_data, label='reference')
    plt.plot(x_data, y_hat, label='hat')
    plt.legend()
    plt.show()
