from typing import Optional, List, Dict, Tuple


class ConfigureCGP:
    def __init__(self,
                 rows: int = 0,
                 cols: int = 0,
                 level_back: int = 0,
                 min_active_num: int = 1,
                 max_active_num: int = 0,
                 funcs: Optional[List[str]] = None,
                 func_in_num: Optional[List[int]] = None,
                 func_red: Optional[List[str]] = None,
                 func_out: Optional[List[str]] = None,
                 weight_ops: Optional[Dict[str, Tuple]] = None):

        """
        Initializes the configuration for a Cartesian Genetic Programming network.

        Args:
            rows (int): Number of rows in the CGP grid.
            cols (int): Number of columns in the CGP grid.
            level_back (int): Level of backward connections allowed.
            min_active_num (int): Minimum number of active nodes.
            max_active_num (int): Maximum number of active nodes.
            funcs (List[str], optional): List of function types for nodes.
            func_in_num (List[int], optional): List of function arities corresponding to each function.
            weight_ops (Dict[List[int]], optional): Dict of weight options.
        """
        self.inputs = 1
        self.functions = funcs if funcs is not None else []
        self.func_in_num = func_in_num if func_in_num is not None else []

        self.reduction_func = func_red

        self.outputs = 1
        self.output_func = func_out
        self.output_inputs = 1

        # CGP grid
        self.rows = rows
        self.cols = cols
        self.level_back = level_back
        self.node_num = self.rows * self.cols
        self.min_active_num = min_active_num
        self.max_active_num = max_active_num

        self.nr_functions = len(self.functions)
        self.output_id = self.nr_functions
        self.nr_functions_out = len(self.output_func)

        self.weight_ops = weight_ops
        self.nr_weights = len(weight_ops)

        self.input_max = max(max(self.nr_functions, 0), self.nr_functions_out)

        self._validate()

    def _validate(self) -> None:
        """Validates the configuration parameters to ensure correct setup."""
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Rows and columns must be positive integers.")

        if not self.functions:
            raise ValueError("Function types (funcs) cannot be empty.")

        if len(self.functions) != len(self.func_in_num):
            raise ValueError("functions and func_in_num must have the same length.")

    def __repr__(self) -> str:
        """Provides a string representation of the configuration."""
        return (
            f"CartesianGPConfigurationW(rows={self.rows}, cols={self.cols}, level_back={self.level_back}, "
            f"min_active_num={self.min_active_num}, max_active_num={self.max_active_num}, "
            f"func_type={self.functions}, func_in_num={self.func_in_num}, "
        )