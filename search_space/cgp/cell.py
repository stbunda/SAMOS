import numpy as np

from genotype import Genotype

class Cell:

    def __init__(self, configuration, reduction, N, R, weights):
        self.configuration = configuration
        self.reduction_config = reduction
        self.normal = N
        self.reduce = R

        self.individual = [Genotype(configuration[i], weights) for i in range(N)]

        self.n_var = self.individual[0].n_var
        self.shape_gen = self.individual[0].gene.shape

    def _insert_reduction(self, node, reduce):
        if reduce:
            node[-1][0] = self.configuration.func_reduction
        else:
            node.pop(-1)
        return node

    def active_net_list(self):
        netlist = []
        start_node = [0 for i in range(self.configuration.input_max + self.configuration.nr_weights)]
        start_node.insert(0, 'input')
        netlist.extend(start_node)

        for n in range(self.normal):
            try:
                self.individual[n].active_net_list()
            except:
                self.individual[n].gene[self.shape_gen[0] - 1][0] = self.configuration[n].out_type_id
                for i in range(self.shape_gen[0] - 2):
                    if self.individual[n].gene[i][0] == self.configuration[n].out_type_id:
                        self.individual[n].gene[i][0] = np.random.randint(len(self.configuration[n].func_type))

        for n in range(self.normal - 1):
            active_block = self.individual[n].active_net_list(len(netlist) - 1 if n > 0 else 0)
            if self.reduction_config[n]:
                active_block[-1][0] = self.configuration.reduction_func
            else:
                active_block.pop(-1)
            netlist.extend(active_block)

        active_block = self.individual[-1].active_net_list(len(netlist) - 1)
        netlist.extend(active_block)

        last_node = [0 for i in range(self.configuration.input_max + self.configuration.nr_weights)]
        last_node[0] = len(netlist) - 1
        last_node.insert(0, self.configuration.output_func)
        netlist.append(last_node)
        return netlist


