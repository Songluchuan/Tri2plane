import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    
    def __init__(self, latent_dim, dims):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.dims = dims
        
        modules = []
        for i in range(len(dims)-1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(dims[-1] * 8*8, latent_dim)

    def encode(self, input):
        z = self.encoder(input)
        z = torch.flatten(z, start_dim=1)
        z = self.fc(z)
        mean = torch.mean(z, 1, keepdim=True)
        var = torch.var(z, 1, keepdim=True)
        z = (z - mean) / var
        return z

    def forward(self, input):
        return self.encode(input)

class NeckDecoder(nn.Module):
    
    def __init__(self, latent_dim, dims):
        super(NeckDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.dims = dims

        self.fc = nn.Linear(latent_dim, dims[0] * 8*8)
        self.upsampler = nn.Sequential(
            nn.Conv2d(dims[0], 4*dims[1], kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(4*dims[1]),
            nn.LeakyReLU(),
            nn.PixelShuffle(2)
        )

    def decode(self, z):
        result = self.fc(z)
        result = result.view(-1, self.dims[0], 8, 8)
        result = self.upsampler(result)
        return result

    def forward(self, input):
        return self.decode(input)

class Decoder(nn.Module):
    
    def __init__(self, dims):
        super(Decoder, self).__init__()

        self.dims = dims

        modules = []
        for i in range(len(dims)-2):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], 4*dims[i+1], kernel_size=3, stride=1, padding=1),
                    nn.InstanceNorm2d(4*dims[i+1]),
                    nn.LeakyReLU(),
                    nn.PixelShuffle(2),
                    nn.Conv2d(dims[i+1], dims[i+1], kernel_size=3, stride=1, padding=1),
                    nn.InstanceNorm2d(dims[i+1]),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(dims[i+1], dims[i+1], kernel_size=3, stride=1, padding=1),
                    nn.InstanceNorm2d(dims[i+1]),
                    nn.LeakyReLU(0.2),
                )
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Conv2d(dims[-2], dims[-1], kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def decode(self, input):
        result = self.decoder(input)
        result = self.final_layer(result)
        return result

    def forward(self, input):
        return self.decode(input)
