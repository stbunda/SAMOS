import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
import torch
from torchmetrics import Accuracy
from functools import partial

import search_space.cell_options as ops


def layer_functions(layer, outputs):
    return partial(layer, *outputs)


class CGPDecoder(pl.LightningModule):
    def __init__(self, netCGP, input_shape, num_classes, AuxHead=False):
        super().__init__()

        self.netCGP = netCGP
        self.AuxHead = AuxHead
        self.encode = []
        self.channels = np.zeros(len(netCGP), dtype=int)
        self.feature_size = np.zeros(len(netCGP), dtype=int)

        self.feature_size[0] = input_shape[0]
        self.channels[0] = input_shape[2]

        default_layers_params = {
            'ConvBlock': (
                ops.ConvBlock, {'input_channels': None, 'output_channels': None, 'kernel_size': None, 'stride': 1}),
            'Bottleneck': (
                ops.Bottleneck, {'input_channels': None, 'output_channels': None, 'kernel_size': None, 'stride': 1}),
            'ResBlock': (
                ops.ResBlock, {'input_channels': None, 'output_channels': None, 'kernel_size': None, 'stride': 1}),
            'SepBlock': (
                ops.SepBlock, {'input_channels': None, 'output_channels': None, 'kernel_size': None, 'stride': 1}),
            'SEBlock': (
                ops.SEBlock, {'input_channels': None}),
            'DilBlock': (
                ops.DilBlock,
                {'input_channels': None, 'output_channels': None, 'kernel_size': None, 'stride': 1, 'dilation': 2}),
            'Conv1x7_7x1': (
                ops.Conv1x7_7x1, {'input_channels': None, 'stride': 1}),
            'MBConvBlock': (
                ops.MBConvBlock, {'input_channels': None, 'output_channels': None, 'kernel_size': None, 'stride': 1}),
            'FusedMBConvBlock': (
                ops.FusedMBConvBlock,
                {'input_channels': None, 'output_channels': None, 'kernel_size': None, 'stride': 1}),
            'AuxiliaryHead': (
                ops.AuxiliaryHead, {'input_channels': None, 'num_classes': num_classes, 'img_size': input_shape[0]}),
        }

        # create model
        # Iterate through the CGP nodes to build the architecture
        for i, (name, node_in1, node_in2, kernel_size, output_channels) in enumerate(netCGP):
            # print(i, (name, node_in1, node_in2, kernel_size, output_channels))

            # First handle the static layer types
            if name == 'input': continue  # Skip input nodes
            if name == 'full':
                self.encode.append(
                    nn.Linear(self.channels[node_in1] * self.feature_size[node_in1] ** 2,
                              num_classes)
                )
                continue

            if 'Pool' in name:
                self.channels[i] = self.channels[node_in1]
                self.feature_size[i] = self.feature_size[node_in1] // 2  # The feature size gets halved due to stride
                if 'Max' in name:
                    self.encode.append(ops.MaxPool(2, stride=2))
                elif 'G' in name:
                    self.feature_size[i] = 1
                    self.encode.append(ops.GAvgPool(1))
                else:
                    self.encode.append(ops.AvgPool(2, stride=2))
                continue

            if name == 'Identity':
                self.channels[i] = self.channels[node_in1]
                self.feature_size[i] = self.feature_size[node_in1]
                self.encode.append(ops.Identity())
                continue

            # Handle Sum and Concat separately due to their unique size updates
            if name == 'Sum':
                # Determine the output channels and size based on inputs
                self.channels[i] = max(self.channels[node_in1], self.channels[node_in2])
                self.feature_size[i] = min(self.feature_size[node_in1], self.feature_size[node_in2])

                # Add the Sum layer
                self.encode.append(ops.Sum())
                continue

            if name == 'Concatenate':
                # Sum the channels from both inputs
                self.channels[i] = self.channels[node_in1] + self.channels[node_in2]

                # Determine the smaller spatial size for alignment
                small_in_id = min(node_in1, node_in2, key=lambda idx: self.feature_size[idx])
                self.feature_size[i] = self.feature_size[small_in_id]

                # Add the Concat layer
                self.encode.append(ops.Concatenate())
                continue

            # Handle the learnable layers types
            layer_class, params = default_layers_params.get(name, (None, None))  # Give default parameters

            if not layer_class:
                raise ValueError(f"Unsupported layer type: {name}")

            if 'input_channels' in params:
                params['input_channels'] = self.channels[node_in1]
            if 'output_channels' in params:
                params['output_channels'] = output_channels
            if 'kernel_size' in params:
                params['kernel_size'] = kernel_size

            # Update the output dimensions for pooling or convolutions
            self.channels[i] = params.get('output_channels', self.channels[node_in1])
            self.feature_size[i] = self.feature_size[node_in1]

            # print(f'\tchannels: {self.channels}')
            # print(f'\tfeature_size: {self.feature_size}')

            # Add the layer
            self.encode.append(layer_class(**params))


        # Create the module list for all layers
        self.module_list = nn.ModuleList(self.encode)
        self.output_array = [None] * len(self.netCGP)

    def forward(self, x):
        aux_output = None
        output_array = self.output_array
        output_array[0] = x
        nodeID = 1

        for layer in self.module_list:
            # Check layer available
            layer_type = type(layer).__name__
            # print(layer_type)
            if layer_type in ops.__all__:
                if layer_type == 'Sum' or layer_type == 'Concatenate':
                    output_array[nodeID] = layer([
                        output_array[self.netCGP[nodeID][1]],
                        output_array[self.netCGP[nodeID][2]]
                    ])
                elif 'Pool' in layer_type:
                    # print(output_array[self.netCGP[nodeID][1]].size())
                    if output_array[self.netCGP[nodeID][1]].size(2) > 1: # and self.AuxHead: # check if the output can be reduced
                        output_array[nodeID] = layer(
                            output_array[self.netCGP[nodeID][1]]
                        )
                    # elif output_array[self.netCGP[nodeID][1]].size(2) > 2 and not self.AuxHead: # check if the output can be reduced
                    #     output_array[nodeID] = layer(
                    #         output_array[self.netCGP[nodeID][1]]
                    #     )
                    else:
                        # print('passing through')
                        output_array[nodeID] = output_array[self.netCGP[nodeID][1]]
                else:
                    output_array[nodeID] = layer(
                        output_array[self.netCGP[nodeID][1]]
                    )
            elif layer_type == 'Linear':
                output_array[nodeID] = layer(
                    output_array[self.netCGP[nodeID][1]].view(output_array[self.netCGP[nodeID][1]].size(0), -1)
                )
            else:
                raise ValueError(f"Error at CGP2CNN forward: Unsupported layer type {type(layer)}.")
            nodeID += 1
            nan = False

            # for name, param in self.module_list.named_parameters():
            #     if torch.isnan(param).any():
            #         print(f"NaN found in {name}")
            #         nan = True
            #     arr = param.cpu().detach().numpy()
            #     if np.any(np.abs(arr) > 1e10):
            #         print(name)
            #         print("Array contains values outside the range [1e-10, 1e10]")


            # if nan:
            #     import matplotlib.pyplot as plt
            #     xi = x.cpu().permute(0, 2, 3, 1).numpy()
            #     fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            #     for i, ax in enumerate(axes.flat):
            #         ax.imshow(xi[i])
            #         ax.axis("off")  # Hide axes for better visualization
            #
            #     plt.tight_layout()
            #     plt.show()
            #
            #     for m in self.module_list[:1]:
            #         for n in m.net:
            #             try:
            #                 print(type(n), n.weight)
            #             except:
            #                 print('no weights')
            #     print('\n')
            # print('\n')
        # values, indices = torch.max(output_array[nodeID-1], dim=1)
        # print(np.row_stack((indices.cpu().detach().numpy(), values.cpu().detach().numpy())))

        # [print(i, np.argmax(b.cpu().numpy())) for i, b in enumerate(output_array[nodeID-1])]
        return output_array[nodeID-1], aux_output