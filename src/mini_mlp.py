import sys
import os

import torch
import torch.nn as nn

import utils
import world


class BatchNormLastDim(nn.Module):
    def __init__(self, s):
        super(BatchNormLastDim, self).__init__()
        self.s = s
        self.bn = nn.BatchNorm1d(s)

    def forward(self, x):
        init_dim = x.shape
        if init_dim[-1] != self.s:
            raise ValueError("batch norm last dim does not have right shape. should be {}, but is {}".format(self.s, init_dim[-1]))

        x_flat = x.view((-1, self.s))
        bn_flat = self.bn(x_flat)
        return bn_flat.view(*init_dim)

class MiniMLP(nn.Sequential):

    def __init__(
        self,
        layer_sizes,
        name='miniMLP',
        activation=nn.ReLU,
        batch_norm=True,
        skip_last_norm=False,
        layer_norm=False,
        dropout=False,
        skip_first_dropout=False,
    ):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):

            is_last = (i+2 == len(layer_sizes))
            
            if dropout:

                if i > 0 or not skip_first_dropout:

                    self.add_module(
                        name + "_mlp_layer_dropout_{:03d}".format(i),
                        nn.Dropout()
                    )


            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                ),
            )

            # Maybe batch_norm
            # (but maybe not on the last layer)
            if batch_norm:
                if (not skip_last_norm) or (not is_last):
                    self.add_module(
                        name + "_mlp_batch_norm_{:03d}".format(i),
                        BatchNormLastDim(layer_sizes[i+1])
                    )
            
            # Maybe layer norm
            # (but maybe not on the last layer)
            if layer_norm:
                if (not skip_last_norm) or (not is_last):
                    self.add_module(
                        name + "_mlp_layer_norm_{:03d}".format(i),
                        nn.LayerNorm(layer_sizes[i+1])
                    )
            
            # Nonlinearity
            # (but not on the last layer)
            if activation is not None and not is_last:
                self.add_module(
                    name + "_mlp_act_{:03d}".format(i),
                    activation()
                )
            
