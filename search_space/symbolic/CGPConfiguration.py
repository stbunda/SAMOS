import numpy as np
from typing import List, Optional


class CartesianGPConfigurationB:
    """
    Configuration for Cartesian Genetic Programming (CGP) used in defining the structure
    of a genetic programming network. This class sets parameters for node arrangement,
    function types, channels, and kernel sizes.
    """

    def __init__(self,
                 rows: int = 0,
                 cols: int = 0,
                 level_back: int = 0,
                 min_active_num: int = 1,
                 max_active_num: int = 0,
                 funcs: Optional[List[str]] = None,
                 funAry: Optional[List[int]] = None,
                 constant_range: Optional[List[int]] = None):
        """
        Initializes the configuration for a Cartesian Genetic Programming network.

        Args:
            rows (int): Number of rows in the CGP grid.
            cols (int): Number of columns in the CGP grid.
            level_back (int): Level of backward connections allowed.
            min_active_num (int): Minimum number of active nodes.
            max_active_num (int): Maximum number of active nodes.
            funcs (List[str], optional): List of function types for nodes.
            funAry (List[int], optional): List of function arities corresponding to each function.
        """
        self.input_num = 1
        self.func_type = funcs if funcs is not None else []
        self.func_in_num = funAry if funAry is not None else []

        self.constant_range = self._initialize_range(constant_range)

        # Set up output configuration
        self.out_num = 1
        self.out_type = ['full']
        self.out_in_num = 1

        # Set CGP grid properties
        self.rows = rows
        self.cols = cols
        self.node_num = self.rows * self.cols
        self.level_back = level_back
        self.min_active_num = min_active_num
        self.max_active_num = max_active_num

        # Function type and arity settings
        self.func_type_num = len(self.func_type)
        self.out_type_id = self.func_type_num
        self.out_type_num = len(self.out_type)

        # Determine the maximum number of inputs per function or output node
        self.max_in_num = max(
            max(self.func_in_num, default=0),
            self.out_in_num
        )

        # Validate configurations
        self._validate_configuration()

    def _initialize_range(self, range: Optional[List[int]]) -> List[int]:
        """Initializes kernel sizes with default values if not provided."""
        return list(np.arange(range[0], range[1] + 1, 1, dtype=int)) if range is not None else list(np.arange(-10, 11, 1, dtype=int))

    def _validate_configuration(self) -> None:
        """Validates the configuration parameters to ensure correct setup."""
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Rows and columns must be positive integers.")

        if not self.func_type:
            raise ValueError("Function types (funcs) cannot be empty.")

        if len(self.func_type) != len(self.func_in_num):
            raise ValueError("funcs and funAry must have the same length.")

    def __repr__(self) -> str:
        """Provides a string representation of the configuration."""
        return (
            f"CartesianGPConfigurationW(rows={self.rows}, cols={self.cols}, level_back={self.level_back}, "
            f"min_active_num={self.min_active_num}, max_active_num={self.max_active_num}, "
            f"func_type={self.func_type}, func_in_num={self.func_in_num}"
        )
