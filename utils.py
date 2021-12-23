import os
import copy
import random
import pickle
import time

import torch
import torchvision
import numpy as np

from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter


class MiniImageNet(torch.utils.data.Dataset):
    """Load MiniImageNet data. Should first put MiniImageNet files as:
        * root:
            |- mini-imagenet-cache-test.pkl 
            |- mini-imagenet-cache-train.pkl
            |- mini-imagenet-cache-val.pkl
        * miniImageNet from: https://github.com/renmengye/few-shot-ssl-public
    """
    def __init__(self, root, mode, n_way, k_shot, k_query, resize):
        """
        Args:
            root: root path of MiniImageNet
            mode: 'training'/'validation'/'test'
            n_way: N-way for support set
            k_shot: K-shot for support set
            k_query: K-shot for query set
            task_size: size of tasks
            startidx: start of index label from
        
        Returns:
        """
        # set input arguments
        self.root = root
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.set_size = self.n_way * self.k_shot
        self.query_size = self.n_way * self.k_query
        self.resize = resize 
        
        # print configuration
        print(f'[INFO] Shuffle dataset: ({str(mode).upper()}), {n_way}-way, {k_shot}-shot, ({k_query}-query-shot)')

        # read raw data suited for each mode (train/test)
        if mode == 'training':
            with open(os.path.join(self.root, 'mini-imagenet-cache-train.pkl'), 'rb') as file:
                data = pickle.load(file)
        elif mode == 'validation':
            with open(os.path.join(self.root, 'mini-imagenet-cache-val.pkl'), 'rb') as file:
                data = pickle.load(file)
        elif mode == 'test':
            with open(os.path.join(self.root, 'mini-imagenet-cache-test.pkl'), 'rb') as file:
                data = pickle.load(file)
        else:
            raise Exception(f'[ERROR] NO mode naemd {str(mode)}... choose among (train/test/validation)')
        
        # define transform module
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
    
        # define dataset length
        self.length = data['image_data'].shape[0]
        
        # tidy up raw data
        proc_data = self.load_split(data)
        self.data = {k: v for k, v in zip([i for i in range(len(proc_data))], list(proc_data.values()))}
        del data, proc_data

        # set total number
        self.num_classes = len(self.data)
        
    def load_split(self, raw_data):
        """Load file list and assign labels.
        Args:
            raw_data: raw input file in the form of dictionary
            
        Returns: 
            dict: {class_name: [image_name_1, image_name_2, ...]}
        """
        # construct dictionary for assigning sample names to each class
        split_dict = {}
        for c, indices in raw_data['class_dict'].items():
            split_dict[c] = raw_data['image_data'][indices[0]:indices[-1] + 1]
        return split_dict

    def __getitem__(self, index):
        """Return set of tasks
        
        Args:
            index: set index
        
        Returns:
        """
        # select n_way classes randomly
        selected_classes = np.random.choice(self.num_classes, self.n_way, False)
        labels = [i for i in range(self.num_classes)]
        class_to_label = dict(zip(selected_classes, labels))

        # select k_shot & k_query from each class
        Sx, Qx = [], []
        Sy, Qy = [], []
        for cls in selected_classes:
            selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
            np.random.shuffle(selected_imgs_idx)

            # get indices for training and test set 
            train_idx = np.array(selected_imgs_idx[:self.k_shot])
            test_idx = np.array(selected_imgs_idx[self.k_shot:])

            # transform each image
            S_temp, Q_temp = [], []
            for idx in range(len(self.data[cls][train_idx])):
                S_temp.append(self.transform(self.data[cls][train_idx][idx]))
                Sy.append(class_to_label[cls])
            for idx in range(len(self.data[cls][test_idx])):
                Q_temp.append(self.transform(self.data[cls][test_idx][idx]))
                Qy.append(class_to_label[cls])
        
            # set support and query set
            Sx.append(torch.stack(S_temp))
            Qx.append(torch.stack(Q_temp))
    
        # tidy up dimension 
        Sx = torch.stack(Sx); Sx = Sx.view(self.n_way * self.k_shot, *Sx.shape[2:])
        Sy = torch.tensor(Sy)
        
        Qx = torch.stack(Qx); Qx = Qx.view(self.n_way * self.k_query, *Qx.shape[2:])
        Qy = torch.tensor(Qy)
        
        # shuffle support set & query set to prevent possible overfit to sample orders 
        S_shuffle_idx = np.random.permutation(len(Sy))
        Sx, Sy = Sx[S_shuffle_idx], Sy[S_shuffle_idx]

        Q_shuffle_idx = np.random.permutation(len(Qy))
        Qx, Qy = Qx[Q_shuffle_idx], Qy[Q_shuffle_idx]
        
        # Sx: (N * K, C, H, W) / Sy: (N * K,) / Qx: (N * Q, C, H, W) / Qy: (N * Q,)
        return Sx, Sy.long(), Qx, Qy.long()

    def __len__(self):
        """Return number of total samples
        Returns: number of total samples
        """
        return self.length
    
def launch_tensor_board(log_path, port, host='0.0.0.0'):
    """Function for initiating TensorBoard.
    
    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True

def initiate_model(model, device):
    """Initiate model instance.
    
    Args:
        model: nn.Module object
        device: torch.device (cpu/cuda)
    
    Returns:
        model: nn.Module instance
    """
    if (torch.cuda.is_available()) and (torch.cuda.device_count() > 1):
        model_instance = torch.nn.DataParallel(model(), device_ids=[i for i in range(torch.cuda.device_count())])
    else:
        model_instance = model()
    model_instance = model_instance.to(device)
    return model_instance
    
def set_weights(model, weights):
    """Function for replace weights of nn.Module instance to new weights.
    
    Args:
        model: model to which weights are applied
        weights: weights to apply

    Returns:
        updated_model
    """
    # make empty containers for storing new weights
    new_dict = copy.deepcopy(model.state_dict())
    new_weights = OrderedDict()
    
    # copy new weights to the containers
    for name, param in weights.named_parameters():
        new_weights[name] = param
    
    # load new weights to the model
    new_dict.update(new_weights)
    model.load_state_dict(new_dict)
    return model

def compute_adapted_weights(weights, gradients, lr):
    """Function to update to adapted weights.
    
    Args:
        weights: weights to be updated
        gradients: gradients required to update the weights
        lr: learning rate for inner loop update
    
    Returns:
        updated_model: nn.Module object
    """
    clone_weights = {k: v.clone() for k, v in weights.items()}
    updated_weights = OrderedDict((name, param - lr * grad) for ((name, param), grad) in zip(clone_weights.items(), gradients))
    return updated_weights

def update_model_weights(model, weights):
    """Function to replace model's weights.
    
    Args:
        model: nn.Module object to replace its weights
        weights: weights for update
    
    Returns:
        updated_model: nn.Module object with newly updated weights
    """
    # define containers for an updated model
    updated_dict = copy.deepcopy(model.state_dict())
    updated_weights = OrderedDict()
    
    # coyp new weights to the containers
    for name, param in weights.items():
        updated_weights[name] = param
    updated_dict.update(updated_weights)
    model.load_state_dict(updated_dict)
    return model

def zero_grad(params):
    """Reset tracked gradients.
    
    Args:
        params: torch.tensor    
    """
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
