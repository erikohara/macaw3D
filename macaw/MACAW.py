"""
Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
Re-implementation by Andrej Karpathy based on https://arxiv.org/abs/1502.03509
"""
import os
import sys

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Laplace, Normal
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.append(os.getcwd())
from .flows import Flow, NormalizingFlowModel
from utils.datasets import CustomDataset


class MACAW:
    """
    The MACAW model.

    Parameters:
    ----------
    config: dict
        A configuration dict that defines all necessary parameters.
        Refer to one of the provided config files for more info/

    Methods:
    ----------
    fit: create a MACAW model from the config and fit it to a predefined causal graph (DAG).
    sample: generate samples from the fitted model
    intervene: Perform an intervention on a given variable in a fitted DAG.
    counterfactual: Answer counterfactual queries on a fitted DAG.

    """

    def __init__(self, config):
        self.c = config
        self.n_layers = config.flow.nl
        self.hidden = config.flow.hm
        self.epochs = config.training.epochs
        self.batch_size = config.training.batch_size
        self.device = config.device

        self.dim = None
        self.model = None
        self.flow_list = None
        self.best_model = None
        self.optimizer = None

        self.pdim = 0
        self.parents = []

    def save_model(self,save_path):
        flow_list_state_dict = []
        for each_flow in self.flow_list:
            flow_list_state_dict.append(each_flow.state_dict())
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_state_dict_flows': flow_list_state_dict
            }, save_path)
    
    def load_model(self,load_path, edges, prior, datashape1):
        self.dim = datashape1
        if type(edges) is tuple:
            print("it worked")
            self.flow_list = []
            for idx_layer in range(self.n_layers):
                if idx_layer % 2 == 0:
                    self.flow_list.append(Flow(self.dim + self.pdim, edges[1], self.device, hm=self.hidden))
                else:
                    self.flow_list.append(Flow(self.dim + self.pdim, edges[0], self.device, hm=self.hidden))
        else:
            self.flow_list = [Flow(self.dim + self.pdim, edges, self.device, hm=self.hidden) for _ in range(self.n_layers)]
        load_dict = torch.load(load_path, map_location=torch.device(self.device))
        for idx_flow, each_flow in enumerate(self.flow_list):
            each_flow.load_state_dict(load_dict['model_state_dict_flows'][idx_flow])
        self.model = NormalizingFlowModel(prior, self.flow_list).to(self.device)
        self.model.load_state_dict(load_dict['model_state_dict'])

    def fit(self, data, edges=None, augment=False):
        """
        Assuming data columns follow the causal ordering, we fit the associated 1D-Equations.

        Parameters:
        ----------
        data: numpy.ndarray
        dag: edges of the predefined causal DAG
        """
        self.dim = data.shape[1]

        if augment:
            data, edges = self._augment(data, edges)

        if self.c.flow.prior_dist == 'laplace':
            prior = Laplace(torch.zeros(self.dim + self.pdim).to(self.device),
                            torch.ones(self.dim + self.pdim).to(self.device))
        elif self.c.flow.prior_dist == 'normal':
            prior = Normal(torch.zeros(self.dim + self.pdim).to(self.device),
                           torch.ones(self.dim + self.pdim).to(self.device))
        else:
            raise Exception("No prior distribution is defined in config!")

        dataset = CustomDataset(data.astype(np.float32), self.device)
        train_loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

        if type(edges) is tuple:
            print("it worked")
            self.flow_list = []
            for idx_layer in range(self.n_layers):
                if idx_layer % 2 == 0:
                    self.flow_list.append(Flow(self.dim + self.pdim, edges[1], self.device, hm=self.hidden))
                else:
                    self.flow_list.append(Flow(self.dim + self.pdim, edges[0], self.device, hm=self.hidden))
        else:
            self.flow_list = [Flow(self.dim + self.pdim, edges, self.device, hm=self.hidden) for _ in range(self.n_layers)]
        self.model = NormalizingFlowModel(prior, self.flow_list).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.c.optim.lr, weight_decay=self.c.optim.weight_decay,
                               betas=(self.c.optim.beta1, 0.999), amsgrad=self.c.optim.amsgrad)

        if self.c.optim.scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=3, verbose=True)
        else:
            scheduler = None

        self.model.train()
        loss_vals = []
        for e in (pbar := tqdm(range(self.epochs))):
            loss_val = 0
            for _, x in enumerate(train_loader):
                x = x.to(self.device)

                _, prior_logprob, log_det = self.model(x)
                loss = - torch.sum(prior_logprob + log_det)
                loss_val += loss.item()

                # optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.c.optim.scheduler:
                scheduler.step(loss_val / len(train_loader))

            pbar.set_description(f'Loss: {loss_val / len(train_loader):.3f}')
            loss_vals.append(loss_val / len(train_loader))

        return loss_vals

    def fit_with_priors(self, data, edges, priors, validation=[], save_path = '', patience=0):
        """
        Assuming data columns follow the causal ordering, we fit the associated 1D-Equations.

        Parameters:
        ----------
        data: numpy.ndarray
        dag: edges of the predefined causal DAG
        """
        self.dim = data.shape[1]

        dataset = CustomDataset(data.astype(np.float32), self.device)

        if len(validation) == 0:

            from torch.utils.data import random_split

            train_size = int(0.9 * len(dataset))
            valid_size = len(dataset) - train_size

            train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
            valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=self.batch_size)
        
        else:
            train_loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)
            valid_dataset = CustomDataset(validation.astype(np.float32), self.device)
            valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=self.batch_size)

        if type(edges) is tuple:
            self.flow_list = []
            for idx_layer in range(self.n_layers):
                if idx_layer % 2 == 0:
                    self.flow_list.append(Flow(self.dim + self.pdim, edges[1], self.device, hm=self.hidden))
                else:
                    self.flow_list.append(Flow(self.dim + self.pdim, edges[0], self.device, hm=self.hidden))
        else:
            self.flow_list = [Flow(self.dim + self.pdim, edges, self.device, hm=self.hidden) for _ in range(self.n_layers)]
        self.model = NormalizingFlowModel(priors, self.flow_list).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.c.optim.lr, weight_decay=self.c.optim.weight_decay,
                               betas=(self.c.optim.beta1, 0.999), amsgrad=self.c.optim.amsgrad)

        if self.c.optim.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=300, eta_min=self.c.optim.lr/100, verbose=False)
        elif self.c.optim.scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=3, verbose=True)
        else:
            scheduler = None

        loss_vals_train = []
        loss_vals_val = []
        lowest_loss_val = 99999999999999
        best_epoch = 0
        epochs_no_improving = 0
        for e in (pbar := tqdm(range(self.epochs))):
            train_loss = 0
            val_loss = 0

            self.model.train()
            for _, x in enumerate(train_loader):
                x = x.to(self.device)

                _, prior_logprob, log_det = self.model(x)
                loss = - torch.sum(prior_logprob + log_det)
                train_loss += loss.item()

                # optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                for _, x in enumerate(valid_loader):
                    x = x.to(self.device)

                    _, prior_logprob, log_det = self.model(x)
                    loss = - torch.sum(prior_logprob + log_det)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(valid_loader)

            if self.c.optim.scheduler == 'cosine':
                scheduler.step()
            elif self.c.optim.scheduler:
                scheduler.step(val_loss)

            pbar.set_description(f'Training Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')
            loss_vals_train.append(train_loss)
            loss_vals_val.append(val_loss)

            if val_loss < lowest_loss_val:
                lowest_loss_val = val_loss
                epochs_no_improving = 0
                best_epoch = e
                if save_path != '':
                    self.save_model(save_path)
            elif patience!=0:
                if epochs_no_improving == patience:
                    print(f"Best epoch was epoch {best_epoch} with a Val losss of {lowest_loss_val:.3f}")
                    return loss_vals_train, loss_vals_val
                else:
                    epochs_no_improving += 1

        print(f"Best epoch was epoch {best_epoch} with a Val losss of {lowest_loss_val:.3f}")
        return loss_vals_train, loss_vals_val

    def sample(self, n_samples):
        self.model.eval()

        with torch.no_grad():
            if type(self.model.priors) == list:
                z = np.zeros((n_samples, self.dim))
                for sl, dist in self.model.priors:
                    z[:, sl] = dist.sample((n_samples,)).cpu().detach().numpy()
            else:
                z = self.model.priors.sample((n_samples,)).cpu().detach().numpy()
            return self._backward_flow(z)[:, self.pdim:]

    def intervene_parent(self, int_vals, n_samples=100):
        """
        We predict the value of x given an intervention do(x_iidx = x0_val)

        This proceeds in 3 steps:
         1) invert flow to find corresponding entry for z_iidx at x_iidx=x0_val
         2) sample z from prior (number of samples is n_samples), and replace z_iidx by inferred value from step 1
         3) propagate z through flow to get samples for x | do(x_iidx=x0_val)
        """
        self.model.eval()
        with torch.no_grad():
            x_int = self.sample(1)
            for key in int_vals:
                x_int[0, key + self.pdim] = int_vals[key]

                if key in self.parents:
                    x_int[0, key] = int_vals[key]

            z_int = self._forward_flow(x_int)
            if type(self.model.priors) == list:
                z = np.zeros((n_samples, self.dim))
                for sl, dist in self.model.priors:
                    z[:, sl] = dist.sample((n_samples,)).cpu().detach().numpy()
            else:
                z = self.model.priors.sample((n_samples,)).cpu().detach().numpy()

            for key in int_vals:
                z[:, key + self.pdim] = z_int[:, key + self.pdim]

                if key in self.parents:
                    z[:, key] = z_int[:, key]

            return self._backward_flow(z)[:, self.pdim:]

    def counterfactual(self, x_obs, cf_vals):
        """
        Given observation x_obs we estimate the counterfactual of setting x_obs[intervention_index] = cf_val

        This proceeds in 3 steps:
         1) abduction - pass-forward through flow to infer latents for x_obs
         2) action - pass-forward again for latent associated with cf_val
         3) prediction - backward pass through the flow
        """
        self.model.eval()
        with torch.no_grad():
            # abduction:
            x_obs = np.hstack([x_obs[:, self.parents]] + [x_obs])
            z_obs = self._forward_flow(x_obs)
            # action (get latent variable value under counterfactual)
            x_cf = np.copy(x_obs)
            for key in cf_vals:
                x_cf[:, key + self.pdim] = cf_vals[key]

                if key in self.parents:
                    x_cf[:, key] = cf_vals[key]

            z_cf_val = self._forward_flow(x_cf)

            for key in cf_vals:
                z_obs[:, key + self.pdim] = z_cf_val[:, key + self.pdim]

                if key in self.parents:
                    z_obs[:, key] = z_cf_val[:, key]
            # prediction (pass through the flow):
            return self._backward_flow(z_obs)[:, self.pdim:]

    def log_likelihood(self, X,idx=0):
        self.model.eval()

        X = np.hstack([X[:, self.parents]] + [X])
        X = torch.tensor(X.astype(np.float32)).to(self.device)
        return self.model.log_likelihood(X,idx)

    def save_best_model(self):
        # save dict the model
        self.model.load_state_dict(self.best_model)
        return self.model

    def _augment(self, data, edges):
        G = nx.DiGraph()
        nodes = np.arange(self.dim).tolist()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        self.parents = [n for n in nodes if len(list(G.predecessors(n))) == 0]
        self.pdim = len(self.parents)

        new_edges = [(i, i + self.pdim) for i in self.parents] + [(i + self.pdim, j + self.pdim) for i, j in edges]
        new_X = np.hstack([data[:, self.parents]] + [data])

        return new_X, new_edges

    def _forward_flow(self, data):
        if self.model is None:
            raise ValueError('Model needs to be fitted first')
        return self.model.forward(torch.tensor(data.astype(np.float32)).to(self.device))[0][-1].detach().cpu().numpy()

    def _backward_flow(self, latent):
        if self.model is None:
            raise ValueError('Model needs to be fitted first')
        return self.model.backward(torch.tensor(latent.astype(np.float32)).to(self.device))[0][
            -1].detach().cpu().numpy()
