import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, img_size=256, img_channels=3, latent_dim=128):
        super(self, Encoder).__init__()

        self.img_size = img_size
        self.img_channels = img_channels
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(self.img_channels, 8, 1),  # (256, 256, 3) -> (256, 256, 8)
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, 3, padding=1),  # (256, 256, 8) -> (256, 256, 16)
        )

    def forward(self, x):

        return self.encoder(x)


class CodeDiscriminator(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=1500):
        super(self, CodeDiscriminator).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.discriminator = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1),  # Sigmoid here?
        )

    def forward(self, x):

        return self.discriminator(x)
