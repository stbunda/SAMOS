import numpy as np


class Genotype:
    def __init__(self, configuration, weights=0):
        self.configuration = configuration
        self.gene_shape = None
        self.weight_columns = weights

        # Calculate total variable size based on configuration
        self.total_nodes = self.configuration.node_num + self.configuration.outputs  # nodes in grid + nr of output nodes
        self.node_size = 1 + self.configuration.input_max + self.nr_weights  # function + nr_connections + weights
        self.n_var = self.total_nodes * self.node_size

        # Initialize integer gene and weight representation for nodes and outputs
        self.gene = np.zeros((self.total_nodes, 1 + self.configuration.max_in_num), dtype=int)
        self.weight = np.zeros((self.total_nodes, self.weight_columns), dtype=int)

        # Initialize active and reduction status arrays
        self.is_active = np.empty(self.total_nodes, dtype=bool)
        self.is_reduction = np.empty(self.total_nodes, dtype=bool)

        self.eval = None

        self.create()

    def create(self):
        rows = self.configuration.rows
        columns = self.configuration.cols
        for n in range(self.total_nodes):
            # Assign function
            if n < self.configuration.node_num:
                nr_functions = self.configuration.nr_functions
                self.gene[n][0] = np.random.randint(nr_functions)
            else:
                self.gene[n][0] = self.configuration.output_id

            # Connection boundaries
            curr_col = np.min((n // rows, columns))
            max_id = curr_col * rows + self.configuration.inputs
            min_id = (curr_col - self.configuration.level_back) * rows + self.configuration.inputs if curr_col - self.configuration.level_back >= 0 else 0

            # Make the connections
            for i in range(1, self.configuration.input_max + 1):
                self.gene[n][i] = min_id + np.random.randint(max_id - min_id)

            # Initialize weights
            for i in range(self.weight_columns):
                self.weight[n][i] = np.random.randint(len(self.configuration.weight_ops[i]))

    def check_active(self):
        self.is_active[:] = False
        for n in range(self.configuration.outputs):
            self._check_path_to_output(self.configuration.node_num + n)

    def _check_path_to_output(self, node_index):
        if not self.is_active[node_index]:
            self.is_active[node_index] = True

            gene_type = self.gene[node_index][0]

            if node_index >= self.configuration.node_num:
                input_count = self.configuration.output_inputs
            else:
                input_count = self.configuration.func_in_num[gene_type]

            for i in range(input_count):
                connected_index = self.gene[node_index][i + 1]
                if connected_index >= self.configuration.inputs:
                    self._check_path_to_output(connected_index - self.configuration.inputs)

    def active_net_list(self, extra=0):
        self.check_active()
