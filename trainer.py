import torch
import copy

import scipy.stats
import numpy as np

from collections import OrderedDict
from tqdm import trange
from torch import nn


from utils import initiate_model

class ProtoTrainer:
    """Trainer module for MAML
    """
    def __init__(self, args, model, device):
        """Initiate MAML trainer
        
        Args:
            model: model instance used for meta-learning
            args: arguments entered by the user
        """
        self.N = args.n_way
        self.K = args.k_spt
        self.Q = args.k_qry
        self.Nt = args.n_way_test
        self.Kt = args.k_spt_test
        self.Qt = args.k_qry_test
        
        self.num_train_points = args.num_train_points
        self.num_test_points = args.num_test_points
        self.dot = args.dot
        
        self.device = device
        self.model = initiate_model(model, self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = args.lr
    
    """
    def __dist__(self, x, y, dim):
        if self.dot:
            return -(x * y).sum(dim)
        else:
            return torch.pow(x - y, 2).sum(dim)
        
    def __batch_dist__(self, P, Q):
        return self.__dist__(P.unsqueeze(0), Q.unsqueeze(1), 2)
    """
    
    def __batch_dist__(self, P, Q):
        if self.dot:
            P_norm = P / P.norm(dim=1)[:, None]
            Q_norm = Q / Q.norm(dim=1)[:, None]
            cos_sim = torch.mm(Q_norm, P_norm.transpose(0, 1))
            return 1. - cos_sim
        else:
            return torch.cdist(Q.unsqueeze(0), P.unsqueeze(0), p=2).squeeze()
    
    def __get_proto__(self, x, y, n_way):
        proto = []
        for label in range(n_way):
            proto.append(torch.mean(x[y==label], 0))
        proto = torch.stack(proto)
        return proto
    
    def train_step(self, dataloader, adjust_lr=False):
        """Method for meta-training
        
        Args:
            dataloader: torch.utils.data.DataLoader object to sample support & query sets from
            adjust_lr: adjustr learning rate (True/False)
            
        Returns:
            te_losses_total: average losses on query set
            te_losses_ci: 95% confidence interval lof losses on query set
            te_accs_total: average accuracies on query set
            te_accs_ci: 95% confidence interval lof accuracies on query set
        """
        # adjust learning rate
        if adjust_lr:
            self.lr *= 0.5
            
        # declare optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # iterative updates
        for _ in trange(self.num_train_points, desc='[INFO] Meta-training', leave=False):
            te_losses, te_accs = 0.0, 0.0
            # get dataset
            Sx, Sy, Qx, Qy = next(iter(dataloader))
            Sx, Sy = Sx.to(self.device).view(self.N * self.K, *Sx.shape[2:]), Sy.to(self.device).view(-1, 1).squeeze()
            Qx, Qy = Qx.to(self.device).view(self.N * self.Q, *Qx.shape[2:]), Qy.to(self.device).view(-1, 1).squeeze()

            # make latent vectors
            Sx = self.model(Sx)
            Qx = self.model(Qx)

            # get prototypes
            S_proto = self.__get_proto__(Sx, Sy, self.N) # n_way x latent_dim

            # get loss on adapted parameter to new tasks (with computational graph attached)
            Qy_hat = (-self.__batch_dist__(S_proto, Qx))
            te_loss = self.criterion(Qy_hat, Qy)

            # update
            optimizer.zero_grad()
            te_loss.backward()
            optimizer.step()
        else:
            # get metric and loss
            te_losses += te_loss.item()

            # get correct counts
            predicted = Qy_hat.argmax(dim=1, keepdim=True)
            te_accs += predicted.eq(Qy.view_as(predicted)).sum().item()
            
            # update metrics for epoch
            te_losses /= self.N * self.Q
            te_accs /= self.N * self.Q
        return te_losses, te_accs

    def eval_step(self, dataloader, mode='val'):
        """Method for meta-testing
        
        Args:
            dataloader: torch.utils.data.DataLoader object to sample support & query sets from
            mode: 'val'/'test'
            
        Returns:
            te_losses_total: average losses on query set
            te_losses_ci: 95% confidence interval lof losses on query set
            te_accs_total: average accuracies on query set
            te_accs_ci: 95% confidence interval lof accuracies on query set
        """     
        # define new model for meta-test
        meta_tester = copy.deepcopy(self.model)
        
        # track losses and metrics
        te_losses_total, te_accs_total = [], []
        
        # iterative updates
        for _ in trange(self.num_train_points if mode=='val' else self.num_test_points, desc='[INFO] Meta-validation' if mode=='val' else '[INFO] Meta-test', leave=False):          
            # track loss and metric per episode
            te_losses, te_accs = 0.0, 0.0
            
            # get dataset
            Sx, Sy, Qx, Qy = next(iter(dataloader))
            Sx, Sy = Sx.to(self.device).view(self.Nt * self.Kt, *Sx.shape[2:]), Sy.to(self.device).view(-1, 1).squeeze()
            Qx, Qy = Qx.to(self.device).view(self.Nt * self.Qt, *Qx.shape[2:]), Qy.to(self.device).view(-1, 1).squeeze()

            # make latent vectors
            Sx = meta_tester(Sx)
            Qx = meta_tester(Qx)

            # get prototypes
            S_proto = self.__get_proto__(Sx, Sy, self.Nt) # n_way x latent_dim

            # get loss on adapted parameter to new tasks (with computational graph attached)
            Qy_hat = (-self.__batch_dist__(S_proto, Qx))
            te_loss = self.criterion(Qy_hat, Qy)

            # get metric and loss
            te_losses += te_loss.item()

            # get correct counts
            predicted = Qy_hat.argmax(dim=1, keepdim=True)
            te_accs += predicted.eq(Qy.view_as(predicted)).sum().item()
            
            # update metrics for epoch
            te_losses /= self.Nt * self.Qt
            te_accs /= self.Nt * self.Qt
    
            te_losses_total.append(te_losses)
            te_accs_total.append(te_accs)
        else:
            # update metrics for epoch
            te_losses_mean = np.asarray(te_losses_total).mean()
            te_accs_mean = np.asarray(te_accs_total).mean()

            # calculate CI constant for collected losses
            te_losses_ci = np.asarray(te_losses_total).std() * np.abs(scipy.stats.t.ppf((1. - 0.95) / 2, len(te_losses_total) - 1)) / np.sqrt(len(te_losses_total))

            # calculate CI constant for accuracies
            te_accs_ci = np.asarray(te_accs_total).std() * np.abs(scipy.stats.t.ppf((1. - 0.95) / 2, len(te_accs_total) - 1)) / np.sqrt(len(te_accs_total))
            return te_losses_mean, te_losses_ci, te_accs_mean, te_accs_ci
