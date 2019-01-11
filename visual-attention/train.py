#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REINFORCE Implementation

[1] Williams, R. J. (1992). 
    Simple statistical gradient-following methods for connectionist reinforcement learning. 
    Machine Learning, 8, 229–256. https://doi.org/10.1007/BF00992696


Created on Mon Jan 15 13:01:27 2018

@author: matthias
@copyright: IOS
"""

import argparse
import sys
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from model import RVA
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from glimpse_sensor import GlimpseSensor

parser = argparse.ArgumentParser(description='PyTorch Recurrent Model for Visual Attention')
#
# TASK PARAMS
parser.add_argument('--mode', type=int, default=0, metavar="",
                    help="0: MNIST, 1: Translated MNIST (default: 0)")
#
# TRAINING PARAMS
parser.add_argument('--epochs', type=int, default=1500, metavar='',
                    help='Amount of epochs for training (default: 100)')
parser.add_argument('--episodes', type=int, default=-1, metavar='',
                    help='Amount of episodes aka mini-batches (default: -1)')
parser.add_argument('--batch_size', type=int, default=100, metavar='',
                    help='Batch size for SGD (default: 100)')
parser.add_argument('--eval_every', type=int, default=10, metavar='',
                    help='Evaluate model in given interval (epochs) (default: 1)')
parser.add_argument('--lrate', type=float, default=0.01, metavar="",
                    help="Learning rate (default: 0.01")
parser.add_argument('--use_hooks', action='store_true', dest='use_hooks',
                    help="Register hooks with debug information (Default: False)")
parser.add_argument('--use_cuda', action='store_true', dest='use_cuda',
                    help="Shall cuda be used (default: False)")
                    
#
# MODEL PARAMS
parser.add_argument('--gamma', type=float, default=1, metavar='',
                   help='Discount factor for reward (default: 0.99)')
parser.add_argument('--location_variance', type=float, default=0.03, metavar='',
                    help='Variance of the estimated gaussian location (default: 0.03)')
parser.add_argument('--glimpse_size', type=int, default=8, metavar='',
                    help='Size of a glimpse in pixels (default: 8)')
parser.add_argument('--glimpse_level', type=int, default=1, metavar='',
                    help='Amount of scale levels of a glimpse (default: 1)')
parser.add_argument('--glimpse_count', type=int, default=6, metavar='',
                    help='Amount of glimpses per example (default: 4)')
parser.add_argument('--clip_grad', type=float, default=10., metavar='',
                    help='Tries to reduce exploding gradient with RNNs (default: 10.)')
parser.add_argument('--use_lstm', action='store_true', dest='use_lstm',
                   help="Shall the core network use LSTM-cells (default: True)")
parser.add_argument('--init_glimpse_rand', type=bool, default=True, metavar="",
                    help="Initialize the first glimpse randomly from [-1, +1] (default:True)")
parser.add_argument('--use_location_rnn', type=bool, default=True, metavar="",
                    help="Shall location policy be learned through rnn (default: True)")
parser.add_argument('--learn_to_scale', action='store_true', dest='learn_to_scale',
                    help="Shall model learn to scale automatically (default: False)") 
parser.add_argument('--gamma_scale', type=float, default=0.005, metavar='',
                    help="The dumping factor of the scale penaltiy (default: 0.1)")
#
# LOG PARAMS
parser.add_argument('--log_every', type=int, default=1000, metavar='',
                    help='Show log output in given interval (episodes) (default: 1000)')
parser.add_argument('--plot', action='store_true', dest='plot',
                    help="Shall python plot intermediate tests with matplotlib (default: False)")
argv = parser.parse_args()
sys.argv = [sys.argv[0]] 


def after_eval_hook(ctx, output0, targets, loss):
     # Print beautiful images
    if argv.plot:
        model.glimpse_sensor.plot_history(num_images=2)
        model.glimpse_sensor.plot_history_full_images(num_images=2)   
    
        lg.info("\nExample inference")
        lg.info("-----------------")
        for tt in range(argv.glimpse_count):
            lg.info("Classification score glimpse # {} True label: {} Location: x: {:01.2f}, y: {:01.2f}, Scale: {}".format(
                    tt, 
                    targets[0], 
                    model.glimpses_location_history[tt][0].item(), 
                    model.glimpses_location_history[tt][0].item(), 
                    model.glimpses_scale_history[tt][0].item())) 
            for cc in range(model.classifier_scores_history[0][0].size(0)):
                lg.info("#{} score: {:.3f}".format(cc, model.classifier_scores_history[tt][0][cc].item()))
        lg.info("\n")
        

from ummon.supervised import ClassificationTrainer
from ummon.logger import Logger
from ummon.trainingstate import Trainingstate
from ummon.modules.loss import VisualAttentionLoss
import ummon.modules.image_transformations as image_transformations

if __name__ == '__main__':
    
    # MNIST
    if argv.mode == 0:
        mnist_data = MNIST("/ext/data/mnist", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
        mnist_data_test = MNIST("/ext/data/mnist", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
   
    # TRANSLATED MNIST
    if argv.mode == 1:    
        transform = transforms.Compose([transforms.ToTensor(), image_transformations.to_translated ])
        mnist_data = MNIST("/ext/data/mnist", train=True, transform=transform, target_transform=None, download=True)
        mnist_data_test = MNIST("/ext/data/mnist", train=False, transform=transform, target_transform=None, download=True)
   
    trainloader = DataLoader(mnist_data, batch_size=argv.batch_size, shuffle=True, sampler=None, batch_sampler=None, num_workers=2)
      
    # INSTANTIATE ENVIRONMENT
    glimpse_sensor = GlimpseSensor(glimpse_size=(argv.glimpse_size, argv.glimpse_size), levels=(argv.glimpse_level, 2), debug=False, optimized=True, enable_history=True)                  
   
    # CHOOSE MODEL
    rva = RVA(args=argv, glimpse_sensor=glimpse_sensor)  
    
    ff = filter(lambda p: p[1].requires_grad == False,  rva.named_parameters())
    print([p for p in ff])
    
    # INSTANTIATE OPTIMIZER
    optimizer = torch.optim.SGD(rva.parameters(), lr=argv.lrate/argv.batch_size, momentum=0.9, nesterov=True)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.5)

    # INTANTIATE LOSS
    criterion = VisualAttentionLoss(rva, argv.gamma, size_average = False)

    # LOAD TRAINING STATE
    try:
        ts = Trainingstate("MNIST_VA_best_validation_loss.pth.tar")
    except FileNotFoundError:
        ts = Trainingstate()
    
    with Logger(loglevel=10, log_batch_interval=500) as lg:
        
        # CREATE A TRAINER
        my_trainer = ClassificationTrainer(lg, 
                            rva, 
                            criterion, 
                            optimizer, 
                            model_filename="MNIST_VA", 
                            precision=np.float32,
                            use_cuda=False,
                            profile=False,
                            trainingstate = ts,
                            after_eval_hook=after_eval_hook)
        
        # START TRAINING
        trainingsstate = my_trainer.fit(dataloader_training=trainloader,
                                    epochs=argv.epochs,
                                    validation_set=mnist_data_test)