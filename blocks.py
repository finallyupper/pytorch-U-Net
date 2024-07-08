import torch as th
import torch.nn as nn

def block1(
        dim_in=1,
        dim_out=64,
    )->nn.Sequential:
        net = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3),
                nn.ReLU(),
                nn.Conv2d(dim_out, dim_out, 3),
                nn.ReLU(),
            #    nn.MaxPool2d(2, 2)
        )
        return net

def block2(
        dim_in
    )->nn.Sequential:
        net = nn.Sequential(
                nn.Conv2d(dim_in, dim_in * 2, 3),
                nn.ReLU(),
                nn.Conv2d(dim_in * 2, dim_in * 2, 3),
                nn.ReLU(),
            #    nn.MaxPool2d(2, 2),
        )
        return net

def up_block1(
        dim_in
)->nn.Sequential:
    net = nn.Sequential(
    #    nn.ConvTranspose2d(dim_in, dim_in / 2, 2), # concat 수행ㅇ
        nn.Conv2d(dim_in, dim_in / 2, 3),
        nn.ReLU(),
        nn.Conv2d(dim_in / 2, dim_in / 2, 3),
        nn.ReLU()
    )

