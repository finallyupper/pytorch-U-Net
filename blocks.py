import torch as th
import torch.nn as nn


def down(
        dim_in,
        dim_out # dim_in * 2
    )->nn.Sequential:
        net = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3),
                nn.ReLU(),
                nn.Conv2d(dim_out, dim_out, 3),
                nn.ReLU(),
            #    nn.MaxPool2d(2, 2),
        )
        return net

def up(
        dim_in,
        dim_out
)->nn.Sequential:
    net = nn.Sequential(
        nn.Conv2d(dim_in, dim_out, 3),
        nn.ReLU(),
        nn.Conv2d(dim_out, dim_out, 3),
        nn.ReLU()
    )

