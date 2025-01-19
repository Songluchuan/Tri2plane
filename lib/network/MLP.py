import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dims, last_op=None):
        super(MLP, self).__init__()

        self.dims = dims
        self.last_op = last_op
        if len(dims) < 5:
            self.skip = None
        else:
            self.skip = int(len(dims) / 2)

        if self.skip:
            layers = []
            for i in range(self.skip - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.LeakyReLU())
            self.layers1 = nn.Sequential(*layers)

            layers = []
            layers.append(nn.Linear(dims[self.skip] + dims[0], dims[self.skip + 1]))
            layers.append(nn.LeakyReLU())
            for i in range(self.skip + 1, len(dims) - 2):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(dims[-2], dims[-1]))
            self.layers2 = nn.Sequential(*layers)
        else:
            layers = []
            for i in range(0, len(dims) - 2):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(dims[-2], dims[-1]))
            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.skip:
            y = self.layers1(x)
            y = torch.cat([y, x], dim=1)
            y = self.layers2(y)
        else:
            y = self.layers(x)
        if self.last_op:
            y = self.last_op(y)
        return y

class OSGDecoder(nn.Module):
    def __init__(self, n_features, out_dim):
        super().__init__()
        self.hidden_dim = 128

        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            torch.nn.Softplus(),
            nn.Linear(self.hidden_dim, out_dim),
        )
        
    def forward(self, sampled_features):
        x = sampled_features
        
        # check shape
        if len(x.shape) == 3:
            N, M, C = x.shape
            x = x.view(N*M, C)

        x = self.net(x)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return rgb, sigma