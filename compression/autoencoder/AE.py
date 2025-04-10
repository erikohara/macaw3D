import math

import numpy as np
import torch
from matplotlib import pyplot as plt

from .nets import Encoder, Decoder


def _o(i, padding, kernel, stride):
    return math.ceil((i + 2 * padding - kernel + 1) / stride)


class AE:
    def __init__(self, encoded_dim, dataloader, lr=0.01, wd = 1e-5):
        self.encoded_dim = encoded_dim

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {self.device}')

        d1, d2 = next(iter(dataloader)).shape[2:]

        o1 = _o(_o(_o(_o(d1, 1, 3, 2), 1, 3, 2), 1, 3, 2), 0, 3, 2)
        o2 = _o(_o(_o(_o(d2, 1, 3, 2), 1, 3, 2), 1, 3, 2), 0, 3, 2)

        self.encoder = Encoder(self.encoded_dim, o1, o2)
        self.decoder = Decoder(self.encoded_dim, o1, o2)
        params = [{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}]
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        self.loss_func = torch.nn.MSELoss()

        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def train(self, dataloader):

        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.encoder.train()
        self.decoder.train()

        train_loss = []
        for image_batch in dataloader:
            image_batch = image_batch.to(self.device)
            encoded_data = self.encoder(image_batch)
            decoded_data = self.decoder(encoded_data)

            loss = self.loss_func(decoded_data, image_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    def test(self, dataloader):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            conc_out = []
            conc_label = []
            for image_batch in dataloader:
                image_batch = image_batch.to(self.device)
                encoded_data = self.encoder(image_batch)
                decoded_data = self.decoder(encoded_data)

                conc_out.append(decoded_data.cpu())
                conc_label.append(image_batch.cpu())

            conc_out = torch.cat(conc_out)
            conc_label = torch.cat(conc_label)
            val_loss = self.loss_func(conc_out, conc_label)
        return val_loss.data

    def encode(self, dataloader):
        self.encoder.eval()

        encoded_data = []
        for image_batch in dataloader:
            image_batch = image_batch.to(self.device)
            with torch.no_grad():
                encoded_data.append(self.encoder(image_batch))

        encoded_data = torch.cat(tuple(encoded_data), 0).cpu().numpy()
        return encoded_data

    def decode(self, dataloader):
        self.decoder.eval()

        decoded_data = []
        for image_batch in dataloader:
            image_batch = image_batch.to(self.device)
            with torch.no_grad():
                decoded_data.append(self.decoder(image_batch))

        decoded_data = torch.cat(tuple(decoded_data), 0).cpu().numpy()
        return decoded_data

    def plot_ae_outputs(self, test_loader, n=10):
        fig = plt.figure(figsize=(20, 4))
        targets = next(iter(test_loader))
        for i in range(n):
            ax = fig.add_subplot(2, n, i + 1)
            img = targets[i].unsqueeze(0).to(self.device)

            self.encoder.eval()
            self.decoder.eval()

            with torch.no_grad():
                rec_img = self.decoder(self.encoder(img))

            ax.imshow(img.cpu().squeeze().numpy(), vmin=0, vmax=1, cmap='gist_gray')
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            if i == n // 2:
                ax.set_title('Original images')

            ax = fig.add_subplot(2, n, i + 1 + n)
            ax.imshow(rec_img.cpu().squeeze().numpy(), vmin=0, vmax=1, cmap='gist_gray')
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            if i == n // 2:
                ax.set_title('Reconstructed images')

        plt.close()
        return fig
