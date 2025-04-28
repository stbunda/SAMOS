import random

import numpy as np
import sympy as sp

unary_operators = {
    'sin': sp.sin,
    'cos': sp.cos,
    'exp': sp.exp,
    'log': lambda v: sp.log(v + 1),  # Avoid log(0)
    'sqrt': sp.sqrt,
    'square': lambda v: v**2,
}

binary_operators = {
    'add': lambda a, b: a + b,
    'mul': lambda a, b: a * b,
    'sub': lambda a, b: a - b,
    'div': lambda a, b: a / (b + 0.01)
}