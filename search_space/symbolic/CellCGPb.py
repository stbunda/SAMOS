import numpy as np
from search_space.symbolic.GenCGPb import GenCgpb

class CellCgpb:

    """
    A class to represent a Cell in Cartesian Genetic Programming (CGP).

    This class manages the configuration, creation, and activation of CGP-based
    neural network cells. Each cell consists of multiple blocks, which are normal
    and reduction blocks in the network's architecture.

    Attributes:
        conf_net (list): List of configurations for each layer in the network.
        reduction (int): Number of reduction blocks.
        normal (int): Number of normal blocks.
        blocks (int): Total number of blocks (normal + reduction).
        individual (list): List of GenCgpW instances representing each block.
        pool (list): Pool configuration for the model.
        n_var_size (int): Number of variables in the genotype of the CGP structure.
        shapegene (tuple): Shape of the gene for the CGP structure.
    """


    def __init__(self, conf_net, N=1):
        """
        Initializes a CellCgpW instance with the specified network configuration.

        Args:
            conf_net (list): Configuration list for each layer in the network.
        """
        self.conf_net = conf_net
        self.normal = N
        self.individual = [
            GenCgpb(conf_net[i], realgene=None, weights=1) for i in range(N)
        ]
        self.n_var_size = self.individual[0].n_var_size
        self.shapegene = self.individual[0].gene.shape

    def active_net_list(self):
        """
        Generates an active network list for the CGP cell with optional auxiliary head.

        Args:
            AuxHead (bool): Flag to add auxiliary head in the network.

        Returns:
            list: List of active nodes in the CGP network.
        """
        finalCGP = []

        for j in range(self.normal):
            try:
                self.individual[j].active_net_list()
            except:
                print(f'Repairing individual, previous fail at index: {j}')
                self.individual[j].gene[self.shapegene[0] - 1][0] = self.conf_net[j].out_type_id
                for i in range(self.shapegene[0] - 2):
                    if self.individual[j].gene[i][0] == self.conf_net[j].out_type_id:
                        self.individual[j].gene[i][0] = np.random.randint(len(self.conf_net[j].func_type))

        n0 = self.individual[0].active_net_list()
        n0.insert(0, ['input', 'x', 0, 0])
        n0.pop(-1)

        finalCGP.extend(n0)

        return finalCGP


