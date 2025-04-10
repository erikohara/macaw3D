import torch
import math
from torch import nn


class Encoder(nn.Module):

    def __init__(self, encoded_dim, o1=3, o2=3):
        super().__init__()

        # Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(64 * o1 * o2, encoded_dim)
            # nn.ReLU(True),
            # nn.Linear(128, encoded_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_dim, o1=3, o2=3):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            # nn.Linear(encoded_dim, 128),
            # nn.ReLU(True),
            nn.Linear(encoded_dim, 64 * o1 * o2),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(64, o1, o2))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1, output_padding=0)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

class Encoder3D(nn.Module):

    def __init__(self, encoded_dim, o0=3, o1=3, o2=3):
        super().__init__()

        # Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv3d(1, 8, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv3d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.Conv3d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.Conv3d(32, 64, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(64 * o0 * o1 * o2, encoded_dim)
            # nn.ReLU(True),
            # nn.Linear(128, encoded_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder3D(nn.Module):

    def __init__(self, encoded_dim, o0=3, o1=3, o2=3):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            # nn.Linear(encoded_dim, 128),
            # nn.ReLU(True),
            nn.Linear(encoded_dim, 64 * o0 * o1 * o2),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(64, o0, o1, o2))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 8, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.ConvTranspose3d(8, 1, 4, stride=2, padding=1, output_padding=0)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

class Encoder3DPauline(nn.Module):

    def __init__(self, encoded_dim, o0=3, o1=3, o2=3):
        super().__init__()

        channel_out = math.ceil(encoded_dim /(o0 * o1 * o2))
        # Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv3d(1,8,3,stride=1,padding=0,dilation=1),
            nn.Conv3d(8,8,3,stride=1,padding=0,dilation=1),
            nn.ReLU(True),
            nn.BatchNorm3d(8),
            nn.MaxPool3d(2,stride=2,padding=1,dilation=1),
            nn.Conv3d(8,16,3,stride=1,padding=0,dilation=1),
            nn.Conv3d(16,16,3,stride=1,padding=0,dilation=1),
            nn.ReLU(True),
            nn.BatchNorm3d(16),
            nn.MaxPool3d(2,stride=2,padding=1,dilation=1),
            nn.Conv3d(16,32,3,stride=1,padding=0,dilation=1),
            nn.Conv3d(32,32,3,stride=1,padding=0,dilation=1),
            nn.ReLU(True),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2,stride=2,padding=1,dilation=1),
            nn.Conv3d(32,channel_out,3,stride=1,padding=0,dilation=1),
            nn.ReLU(True)
        )

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(channel_out * o0 * o1 * o2, encoded_dim)
            # nn.ReLU(True),
            # nn.Linear(128, encoded_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder3DPauline(nn.Module):

    def __init__(self, encoded_dim, o0=3, o1=3, o2=3):
        super().__init__()

        channel_out = math.ceil(encoded_dim /(o0 * o1 * o2))

        self.decoder_lin = nn.Sequential(
            # nn.Linear(encoded_dim, 128),
            # nn.ReLU(True),
            nn.Linear(encoded_dim, channel_out * o0 * o1 * o2),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(channel_out, o0, o1, o2))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(channel_out, 32, 3, stride=1, padding=1, dilation=1),
            nn.ConvTranspose3d(32, 32, 3, stride=1, padding=0, dilation=1),
            nn.ReLU(True),
            nn.BatchNorm3d(32),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.ConvTranspose3d(32, 16, 3, stride=1, padding=1, dilation=1),
            nn.ConvTranspose3d(16, 16, 3, stride=1, padding=0, dilation=1),
            nn.ReLU(True),
            nn.BatchNorm3d(16),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.ConvTranspose3d(16, 8, 3, stride=1, padding=1, dilation=1),
            nn.ConvTranspose3d(8, 8, 3, stride=1, padding=0, dilation=1),
            nn.ReLU(True),
            nn.BatchNorm3d(8),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.ConvTranspose3d(8, 1, 3, stride=1, padding=0, dilation=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

class ResBlock3d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv3d(dim, dim, 3, 1, 1),
            nn.BatchNorm3d(dim),
            nn.ReLU(True),
            nn.Conv3d(dim, dim, 1),
            nn.BatchNorm3d(dim)
        )

    def forward(self, x):
        y = x.copy() + self.block(x)
        return y