import torch
import torch.nn as nn

from .functions import vq, vq_st


def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512, image_dim=2):
        super().__init__()

        self.image_dim = image_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim, image_dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.encoder3d = nn.Sequential(
            nn.Conv3d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm3d(dim),
            nn.ReLU(True),
            nn.Conv3d(dim, dim, 5, 3, 1),
            ResBlock3d(dim),
            ResBlock3d(dim),
        )

        self.decoder3d = nn.Sequential(
            ResBlock3d(dim),
            ResBlock3d(dim),
            nn.ReLU(True),
            nn.ConvTranspose3d(dim, dim, 5, 3, 1),
            nn.BatchNorm3d(dim),
            nn.ReLU(True),
            nn.ConvTranspose3d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        if self.image_dim == 3:
            z_e_x = self.encoder3d(x)
        else:
            z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents
    
    def encode_without_codebook(self, x):
        if self.image_dim == 3:
            latents = self.encoder3d(x)
        else:
            latents = self.encoder(x)
        #latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        print(f"latents.shape: {latents.shape}")
        print(f"self.codebook.embedding(latents).shape : {self.codebook.embedding(latents).shape}")
        if self.image_dim == 3:
            z_q_x = self.codebook.embedding(latents).permute(0, 4, 1, 2, 3)  # (B, D, HxWxL)
        else:
            z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        if self.image_dim == 3:
            x_tilde = self.decoder3d(z_q_x)
        else:
            x_tilde = self.decoder(z_q_x)
        return x_tilde

    def decode_with_codebook(self, z_e_x):
        latents = self.codebook(z_e_x)
        if self.image_dim == 3:
            z_q_x = self.codebook.embedding(latents).permute(0, 4, 1, 2, 3)  # (B, D, HxWxL)
        else:
            z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        if self.image_dim == 3:
            x_tilde = self.decoder3d(z_q_x)
        else:
            x_tilde = self.decoder(z_q_x)
        return x_tilde
    
    def forward(self, x):
        if self.image_dim == 3:
            z_e_x = self.encoder3d(x)
        else:
            z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        if self.image_dim == 3:
            x_tilde = self.decoder3d(z_q_x_st)
        else:
            x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

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
        return x + self.block(x)


class VQEmbedding(nn.Module):
    def __init__(self, K, D, image_dim):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1. / K, 1. / K)
        self.image_dim = image_dim

    def forward(self, z_e_x):
        if self.image_dim == 3:
            z_e_x_ = z_e_x.permute(0, 2, 3, 4, 1).contiguous()
        else:
            z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        if self.image_dim == 3:
             z_e_x_ = z_e_x.permute(0, 2, 3, 4, 1).contiguous()
        else:
            z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        if self.image_dim == 3:
            z_q_x = z_q_x_.permute(0, 4, 1, 2, 3).contiguous()
        else:
            z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
                                               dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        if self.image_dim == 3:
            z_q_x_bar = z_q_x_bar_.permute(0, 4, 1, 2, 3).contiguous()
        else:
            z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar
