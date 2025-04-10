import math

import numpy as np
import torch
from matplotlib import pyplot as plt

from .nets import Encoder3D, Decoder3D


def _o(i, padding, kernel, stride):
    return math.ceil((i + 2 * padding - kernel + 1) / stride)


class AE3D:
    def __init__(self, encoded_dim, dataloader, lr=0.01):
        self.encoded_dim = encoded_dim

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {self.device}')
        images = next(iter(dataloader))
        d0, d1, d2 = images.shape[2:]

        o0 = _o(_o(_o(_o(d0, 1, 3, 2), 1, 3, 2), 1, 3, 2), 0, 3, 2)
        o1 = _o(_o(_o(_o(d1, 1, 3, 2), 1, 3, 2), 1, 3, 2), 0, 3, 2)
        o2 = _o(_o(_o(_o(d2, 1, 3, 2), 1, 3, 2), 1, 3, 2), 0, 3, 2)

        print(f"Show sizes after conv: {o0}, {o1}, {o2}")

        self.encoder = Encoder3D(self.encoded_dim, o0, o1, o2)
        self.decoder = Decoder3D(self.encoded_dim, o0, o1, o2)

        params = [{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}]
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-05)
        self.loss_func = torch.nn.MSELoss()

        self.encoder.to(self.device)
        self.decoder.to(self.device)
    
    def save_checkpoint(self,save_path,current_epoch,best_val_loss,losses):
        torch.save({
            'epoch': current_epoch,
            'model_state_dict_encoder': self.encoder.state_dict(),
            'model_state_dict_decoder': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_func,
            'best_val_loss': best_val_loss,
            'losses': losses
            }, save_path)

    def load_checkpoint(self,load_path):
        checkpoint = torch.load(load_path, map_location=torch.device(self.device))
        self.encoder.load_state_dict(checkpoint['model_state_dict_encoder'])
        self.decoder.load_state_dict(checkpoint['model_state_dict_decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        self.loss_func = checkpoint['loss']
        losses = checkpoint['losses']
        best_val_loss = checkpoint['best_val_loss']
        return epoch, losses, best_val_loss

    def train(self, dataloader):

        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.encoder.train()
        self.decoder.train()

        train_loss = []
        for image_batch in dataloader:
            #print(f"dataloading {len(train_loss)}")
            #image_batch, _, _, _, _ = one_batch
            image_batch = image_batch.to(self.device)
            encoded_data = self.encoder(image_batch)
            decoded_data = self.decoder(encoded_data)
            loss = self.loss_func(decoded_data, image_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            '''
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
            '''

        return np.sum(train_loss)/len(train_loss)

    def test(self, dataloader):
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            #conc_out = []
            #conc_label = []
            #first = True
            val_loss = []
            for image_batch in dataloader:
                #image_batch, _, _, _, _ = one_batch
                image_batch = image_batch.to(self.device)
                encoded_data = self.encoder(image_batch)
                decoded_data = self.decoder(encoded_data)
                loss = self.loss_func(decoded_data, image_batch)
                val_loss.append(loss.item())
                '''
                if not first:
                    conc_out = torch.cat((conc_out,decoded_data),0)
                    conc_label = torch.cat((conc_label,image_batch),0)
                else:
                    first = False
                    conc_out = decoded_data
                    conc_label = image_batch
                '''
                #conc_out.append(decoded_data.cpu())
                #conc_label.append(image_batch.cpu())
        return np.sum(val_loss)/len(val_loss)

    def encode(self, dataloader):
        self.encoder.eval()

        encoded_data = []
        #all_eid = []
        for image_batch in dataloader:
            #image_batch, sex, age, bmi, eid = one_batch
            image_batch = image_batch.to(self.device)
            with torch.no_grad():
                encoded_data.append(self.encoder(image_batch))
                #all_eid.append(list(eid))

        encoded_data = torch.cat(tuple(encoded_data), 0).cpu().numpy()
        #all_eid = sum(all_eid, [])
        return encoded_data #, all_eid

    def decode(self, dataloader):
        self.decoder.eval()

        decoded_data = []
        for image_batch in dataloader:
            image_batch = image_batch.to(self.device)
            with torch.no_grad():
                decoded_data.append(self.decoder(image_batch).cpu().numpy())

        decoded_data = np.concatenate(decoded_data, 0)
        return decoded_data

    def plot_ae_outputs(self, test_loader, n=10, slice=50):
        fig = plt.figure(figsize=(20, 4))
        targets = next(iter(test_loader))
        for i in range(n):
            ax = fig.add_subplot(2, n, i + 1)
            img = targets[i].unsqueeze(0).to(self.device)

            self.encoder.eval()
            self.decoder.eval()

            with torch.no_grad():
                rec_img = self.decoder(self.encoder(img))

            ax.imshow(img[:,:,slice].cpu().squeeze().numpy(), vmin=0, vmax=1, cmap='gist_gray')
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            if i == n // 2:
                ax.set_title('Original images')

            ax = fig.add_subplot(2, n, i + 1 + n)
            ax.imshow(rec_img[:,:,slice].cpu().squeeze().numpy(), vmin=0, vmax=1, cmap='gist_gray')
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            if i == n // 2:
                ax.set_title('Reconstructed images')

        plt.close()
        return fig
