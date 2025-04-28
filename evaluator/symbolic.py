from typing import Dict, List

import numpy as np
from pymoo.core.problem import Problem

from search_space.symbolic.symbolic_regression import CGPDecoder
import sympy as sp


class MO_evaluation(Problem):
    def __init__(
            self,
            objectives: Dict = None,
            nvar_real: int = 0,
            x_data = None,
            y_data = None,

    ):
        super().__init__(n_var=1, n_obj=len(objectives), n_constr=0, requires_kwargs=True)
        self.objectives = objectives
        self.x_data = x_data
        self.y_data = y_data
        self.problem_type = 'symbolic'

        # Bounds for real-valued variables
        self.xl = np.zeros(nvar_real)
        self.xu = 0.999999 * np.ones(nvar_real)

    def calc_mse(self, y_pred):
        mse = np.mean((y_pred - self.y_data) ** 2)
        # mse = np.log10(mse)
        if np.isnan(mse):
            return -1
        elif mse > 1e6:
            return -1
        else:
            return mse

    def calc_complexity(self, sym):
        return len(sym.netCGP)

    def _evaluate(self, x, out, *args, **kwargs):
        all_metrics = []
        for m, model_instance in enumerate(x):
            X = sp.Symbol('x')
            sym = CGPDecoder(model_instance[0].active_net_list())
            try:
                y_pred = np.array([sym.net.subs(X, val) for val in self.x_data], dtype=np.float64)
            except:
                y_pred = np.ones_like(self.x_data) * np.nan


            # Compute error
            metrics = []
            if "mse" in self.objectives:  # Mean Squared Error
                mse = self.calc_mse(y_pred)

                metrics.append(mse)
            if "mae" in self.objectives:  # Mean Absolute Error
                metrics.append(np.mean(np.abs(y_pred - self.y_data)))
            if 'complexity' in self.objectives:
                metrics.append(self.calc_complexity(sym))
            all_metrics.append(metrics)
            print(m, metrics, sym.net)
        out['F'] = np.array(all_metrics, dtype=float)




