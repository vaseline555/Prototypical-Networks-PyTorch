import os
import sys
import time
import json
import random
import argparse
import threading
import numpy as np

import torch
import torch.nn as nn
from multiprocessing import cpu_count
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from trainer import ProtoTrainer
from utils import MiniImageNet, launch_tensor_board
from models import ProtoConvNet

def main(args, writer, filename):
    """Main program to run prototypical networks (ProtoNet).
    
    Args:
        args: user input arguments parsed by argparser
        writer: SummaryWriter instance for TensorBoard tracking
        filename: file name for results and models
    
    Retunrs:
    """
    # set seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # print arguments
    print(args)
    
    # set device (CPU/GPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # define model
    model = ProtoConvNet
        
    # define trainer
    proto_trainer = ProtoTrainer(args, model, device)
    
    # define dataset for meta-training
    meta_train_dataset = MiniImageNet(
        root=args.data_path, mode='training',
        n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry,
        resize=args.image_size
    )
    
    # define dataset for meta-testing
    meta_val_dataset = MiniImageNet(
        root=args.data_path, mode='validation',
        n_way=args.n_way_test, k_shot=args.k_spt_test, k_query=args.k_qry_test,
        resize=args.image_size
    )
    
    # define dataset for meta-testing
    meta_test_dataset = MiniImageNet(
        root=args.data_path, mode='test',
        n_way=args.n_way_test, k_shot=args.k_spt_test, k_query=args.k_qry_test,
        resize=args.image_size
    )   
        
    # define dataloader for meta-train
    meta_train_dataloader = torch.utils.data.DataLoader(
        dataset=meta_train_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True
    )
    
    # define dataloader for meta-validation
    meta_val_dataloader = torch.utils.data.DataLoader(
        dataset=meta_val_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True
    )
    
    # define dataloader for meta-test
    meta_test_dataloader = torch.utils.data.DataLoader(
        dataset=meta_test_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True
    )

    # result container
    results = defaultdict(list)
    
    # track validation accuracy for earlystopping
    best_val, count = 0.0, 0
    
    # do meta-learning
    for epoch in trange(args.epoch):
        # meta-training
        if (epoch * args.num_train_points) % args.lr_step == 0:
            te_losses, te_accs = proto_trainer.train_step(meta_train_dataloader, True)
        else:
            te_losses, te_accs = proto_trainer.train_step(meta_train_dataloader, False)
        
        # add to TensorBoard
        writer.add_scalars(
                'Meta-training Loss',
                {f"{args.n_way}-{args.k_spt} Meta-training Loss_{args.seed}": te_losses},
                epoch + 1
                )
        writer.add_scalars(
                'Meta-training Accuracy',
                {f"{args.n_way}-{args.k_spt} Meta-training Accuracy_{args.seed}": te_accs},
                epoch + 1
                )
        
        # add to result container
        results['mtr_loss'].append(te_losses)
        results['mtr_acc'].append(te_accs)
        
        # meta-validation
        if epoch % 1 == 0:
            te_losses_mean, te_losses_ci, te_accs_mean, te_accs_ci = proto_trainer.eval_step(meta_val_dataloader, 'val')
            
            # check earlystopping criterion
            if best_val < te_accs_mean:
                best_val = te_accs_mean
                count = 0
            else:
                count += 1
                
            # add to TensorBoard
            writer.add_scalars(
                    'Meta-validaton Loss',
                    {f"{args.n_way}-{args.k_spt} Meta-validaton Loss_{args.seed}": te_losses_mean},
                    epoch // 1 + 1
                    )
            writer.add_scalars(
                    'Meta-validaton Accuracy',
                    {f"{args.n_way}-{args.k_spt} Meta-validaton Accuracy_{args.seed}": te_accs_mean},
                    epoch // 1 + 1
                    )

            # add to result container
            results['mval_loss'].append(te_losses_mean); results['mval_loss_ci'].append(te_losses_ci)
            results['mval_acc'].append(te_accs_mean); results['mval_acc_ci'].append(te_accs_ci)
            
            # save model
            torch.save(proto_trainer.model, os.path.join(os.path.join(args.model_path, filename), f'{args.n_way}-{args.k_spt}_dot_{args.dot}_{args.n_way_test}_{args.k_spt_test}_{args.seed}_checkpoint_{str(epoch + 1).zfill(5)}.pt'))

            # save result container as JSON format
            with open(os.path.join(os.path.join(args.result_path, filename), f'{args.n_way}-{args.k_spt}_dot_{args.dot}_{args.n_way_test}_{args.k_spt_test}_{args.seed}_results_{str(epoch + 1).zfill(5)}.json'), 'w') as file:
                json.dump(results, file)
            
            # do earlystopping
            if count == 15:
                if epoch > 100:
                    # meta-testing
                    te_losses_mean, te_losses_ci, te_accs_mean, te_accs_ci = proto_trainer.eval_step(meta_test_dataloader, 'test')

                    # add to TensorBoard
                    writer.add_scalars(
                            'Meta-test Loss',
                            {f"{args.n_way}-{args.k_spt} Meta-test Loss_{args.seed}": te_losses_mean},
                            args.epoch
                            )
                    writer.add_scalars(
                            'Meta-test Accuracy',
                            {f"{args.n_way}-{args.k_spt} Meta-test Accuracy_{args.seed}": te_accs_mean},
                            args.epoch
                            )

                    # add to result container
                    results['mte_loss'].append(te_losses_mean); results['mte_loss_ci'].append(te_losses_ci)
                    results['mte_acc'].append(te_accs_mean); results['mte_acc_ci'].append(te_accs_ci)

                    # save model
                    torch.save(proto_trainer.model, os.path.join(os.path.join(args.model_path, filename), f'{args.n_way}-{args.k_spt}_dot_{args.dot}_{args.n_way_test}_{args.k_spt_test}_{args.seed}_checkpoint_{str(args.epoch).zfill(5)}.pt'))

                    # save result container as JSON format
                    with open(os.path.join(os.path.join(args.result_path, filename), f'{args.n_way}-{args.k_spt}_dot_{args.dot}_{args.n_way_test}_{args.k_spt_test}_{args.seed}_results_{str(args.epoch).zfill(5)}.json'), 'w') as file:
                        json.dump(results, file)
                    break
                else:
                    count = 0
    else:
        # meta-testing
        te_losses_mean, te_losses_ci, te_accs_mean, te_accs_ci = proto_trainer.eval_step(meta_test_dataloader, 'test')
            
        # add to TensorBoard
        writer.add_scalars(
                'Meta-test Loss',
                {f"{args.n_way}-{args.k_spt} Meta-test Loss_{args.seed}": te_losses_mean},
                args.epoch
                )
        writer.add_scalars(
                'Meta-test Accuracy',
                {f"{args.n_way}-{args.k_spt} Meta-test Accuracy_{args.seed}": te_accs_mean},
                args.epoch
                )

        # add to result container
        results['mte_loss'].append(te_losses_mean); results['mte_loss_ci'].append(te_losses_ci)
        results['mte_acc'].append(te_accs_mean); results['mte_acc_ci'].append(te_accs_ci)
   
        # save model
        torch.save(proto_trainer.model, os.path.join(os.path.join(args.model_path, filename), f'{args.n_way}-{args.k_spt}_dot_{args.dot}_{args.n_way_test}_{args.k_spt_test}_{args.seed}_checkpoint_{str(args.epoch).zfill(5)}.pt'))

        # save result container as JSON format
        with open(os.path.join(os.path.join(args.result_path, filename), f'{args.n_way}-{args.k_spt}_dot_{args.dot}_{args.n_way_test}_{args.k_spt_test}_{args.seed}_results_{str(args.epoch).zfill(5)}.json'), 'w') as file:
            json.dump(results, file)


if __name__ == '__main__':
    # parse user inputs as arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to read data from', default='./data')
    parser.add_argument('--model_path', type=str, help='path to save a model', default='./model')
    parser.add_argument('--log_path', type=str, help='path to save log to', default='./log')
    parser.add_argument('--result_path', type=str, help='path to save log to', default='./result')
    parser.add_argument('--tb_port', type=int, help='TensorBoard port number', default=6006)
    parser.add_argument('--seed', type=int, help='random seed', default=5959)
    parser.add_argument('--epoch', type=int, help='epoch number', default=300)
    parser.add_argument('--n_way', '-N', type=int, help='N-way', default=5)
    parser.add_argument('--k_spt', '-K', type=int, help='K-shot for support set', default=1)
    parser.add_argument('--k_qry', '-Q', type=int, help='K-shot for query set', default=15)
    parser.add_argument('--n_way_test', '-Nt', type=int, help='N-way for test set', default=5)
    parser.add_argument('--k_spt_test', '-Kt', type=int, help='K-shot for support set in test set', default=1)
    parser.add_argument('--k_qry_test', '-Qt', type=int, help='K-shot for query set in test set', default=15)
    parser.add_argument('--image_size', type=int, help='specify image size, if smaller than original size, it will be resized', default=84)
    parser.add_argument('--num_train_points', type=int, help='number of training episodes to make final results', default=100)
    parser.add_argument('--num_test_points', type=int, help='number of test episodes to make final results', default=600)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('--lr_step', type=float, help='adjust learning rate at every `lr_step` episodes', default=2000)
    parser.add_argument('--dot', action='store_true', help='use cosine distance')
    args = parser.parse_args()
    
    # check data directory exists
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f'[ERROR] Data is NOT prepared at {args.data_path}!')
    
    filename = f'{args.n_way}-{args.k_spt}_dot_{args.dot}_{args.n_way_test}_{args.k_spt_test}_{str(args.seed)}'
    
    # check if model path exists, and make if it is not
    if not os.path.isdir(os.path.join(args.model_path, filename)):
        os.makedirs(os.path.join(args.model_path, filename))
                       
    # check if log path exists, and make if it is not
    if not os.path.isdir(os.path.join(args.log_path, filename)):
        os.makedirs(os.path.join(args.log_path, filename))
    
    # check if result path exists, and make if it is not
    if not os.path.isdir(os.path.join(args.result_path, filename)):
        os.makedirs(os.path.join(args.result_path, filename))
    
    # run TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.log_path, filename), filename_suffix="ProtoNet")
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([os.path.join(args.log_path, filename), args.tb_port, '0.0.0.0'])
        ).start()
    time.sleep(3.0)
    
    # run main program
    main(args, writer, filename)
    
    # done!
    print('[INFO] ...done experiments!')
    sys.exit(0)
