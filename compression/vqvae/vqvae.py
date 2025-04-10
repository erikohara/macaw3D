import numpy as np
import torch
import torch.nn.functional as F

from .modules import VectorQuantizedVAE


class VQVAE:
    def __init__(self, config, writer):
        self.config = config
        self.device = config.device
        self.writer = writer

        self.model = VectorQuantizedVAE(config.vq.num_channels, 
                                        config.vq.hidden_size, 
                                        config.vq.k, 
                                        image_dim=config.vq.image_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.optim.lr)
    
    def save_checkpoint(self,save_path,current_epoch,best_val_loss,losses):
        torch.save({
            'epoch': current_epoch,
            'model': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'losses': losses
            }, save_path)

    def load_checkpoint(self,load_path):
        checkpoint = torch.load(load_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        best_val_loss = checkpoint['best_val_loss']
        return epoch, losses, best_val_loss

    def generate_samples(self, images):
        with torch.no_grad():
            images = images.to(self.device)
            x_tilde, _, _ = self.model(images)
        return x_tilde

    def train(self, data_loader):
        losses_recons, losses_vq = 0., 0.
        for images in data_loader:
            images = images.to(self.device)

            self.optimizer.zero_grad()
            x_tilde, z_e_x, z_q_x = self.model(images)

            # Reconstruction loss
            loss_recons = F.mse_loss(x_tilde, images)
            # Vector quantization objective
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
            # Commitment objective
            loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

            losses_recons += loss_recons
            losses_vq += loss_vq
            loss = loss_recons + loss_vq + self.config.optim.beta * loss_commit
            loss.backward()

            # Logs
            self.writer.add_scalar('loss/train/reconstruction', loss_recons.item(), self.config.steps)
            self.writer.add_scalar('loss/train/quantization', loss_vq.item(), self.config.steps)

            self.optimizer.step()
        
        losses_recons /= len(data_loader)
        losses_vq /= len(data_loader)
        return losses_recons.item(), losses_vq.item()

    def test(self, data_loader):
        with torch.no_grad():
            loss_recons, loss_vq = 0., 0.
            for images in data_loader:
                images = images.to(self.device)
                x_tilde, z_e_x, z_q_x = self.model(images)
                loss_recons += F.mse_loss(x_tilde, images)
                loss_vq += F.mse_loss(z_q_x, z_e_x)

            loss_recons /= len(data_loader)
            loss_vq /= len(data_loader)

        # Logs
        self.writer.add_scalar('loss/test/reconstruction', loss_recons.item(), self.config.steps)
        self.writer.add_scalar('loss/test/quantization', loss_vq.item(), self.config.steps)

        return loss_recons.item(), loss_vq.item()

    def encode(self, data_loader):

        latents = []
        with torch.no_grad():
            for images in data_loader:
                l = self.model.encode(images.to(self.config.device))
                latents.append(l.cpu().numpy().reshape(images.shape[0], -1))

        return np.vstack(latents)

    def encode_without_codebook(self, data_loader):

        latents = []
        with torch.no_grad():
            for images in data_loader:
                l = self.model.encode_without_codebook(images.to(self.config.device))
                #last_dim = images.shape[2] * images.shape[3] * images.shape[4]
                #latents.append(l.cpu().numpy().reshape((images.shape[0], images.shape[1], last_dim)))
                latents.append(l.cpu().numpy())
        
        print(f"latents[0].shape: {latents[0].shape}")
        latents = np.concatenate(latents,axis=0)
        #last_dim = latents.shape[2] * latents.shape[3] * latents.shape[4]
        return latents.reshape((latents.shape[0],latents.shape[1],-1))

    def decode(self, data):
        with torch.no_grad():
            x_t = self.model.decode(torch.tensor(data).to(self.config.device))
            return np.squeeze(x_t.cpu().numpy())
        
    def decode_with_codebook(self, data):
        with torch.no_grad():
            x_t = self.model.decode_with_codebook(torch.tensor(data).to(self.config.device))
            return np.squeeze(x_t.cpu().numpy())

