#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch GMVAE')

# TRAINING PARAMS
parser.add_argument('--epochs', type=int, default=1500, metavar='',
                    help='Amount of epochs for training (default: 100)')
parser.add_argument('--batch_size', type=int, default=100, metavar='',
                    help='Batch size for SGD (default: 100)')
parser.add_argument('--lrate', type=float, default=0.01, metavar="",
                    help="Learning rate (default: 0.01")
parser.add_argument('--use_cuda', action='store_true', dest='use_cuda',
                    help="Shall cuda be used (default: False)")
argv = parser.parse_args()
sys.argv = [sys.argv[0]] 

from ummon import *
import ummon.utils as uu
from ummon.unsupervised import UnsupervisedTrainer
from ummon.logger import Logger
from ummon.trainingstate import Trainingstate
from gmvae import GMVAE
from gmvae import NegVariationalLowerBound

torch.manual_seed(4)

if __name__ == '__main__':
    
    def binarize(x, double = False):
        '''
        Binarize Image
        '''
        m = torch.distributions.Uniform(0, 1)
        xb = m.sample(x.size())
        bin_image = (x > xb).float() * 1
        if double == True:
            return bin_image.double()
        else:
            return bin_image
    
    # MNIST
    transform = transforms.Compose([transforms.ToTensor(), binarize])
    mnist_data = MNIST("/ext/data/mnist", train=True, transform=transform, target_transform=None, download=True)
    mnist_data_test = MNIST("/ext/data/mnist", train=False, transform=transform, target_transform=None, download=True)
   
    trainloader = DataLoader(mnist_data, batch_size=argv.batch_size, shuffle=True, sampler=None, batch_sampler=None, num_workers=2)
      
    # MODEL
    gmvae = GMVAE()                 
    
    # INTANTIATE LOSS
    criterion = NegVariationalLowerBound(gmvae, size_average=False)
  
    # INSTANTIATE OPTIMIZER
    optimizer = torch.optim.Adam(gmvae.parameters())
    
    # LOAD TRAINING STATE
    try:
        ts = Trainingstate("MNIST_GMVAE.pth.tar")
    except FileNotFoundError:
        ts = Trainingstate()
    
    with Logger(loglevel=10, logdir='.', log_batch_interval=20) as lg:

        # EARLY STOPPING
        earlystop = StepLR_earlystop(optimizer, ts, gmvae, step_size=100, nsteps=5, patience=20, logger=lg)
        
        # CREATE A TRAINER
        my_trainer = UnsupervisedTrainer(lg, 
                            gmvae, 
                            criterion, 
                            optimizer, 
                            scheduler = earlystop,
                            trainingstate = ts,
                            model_filename="MNIST_GMVAE",
                            use_cuda= argv.use_cuda,
                            after_eval_hook = criterion.compute_special_losses)
        
        # START TRAINING
        trainingsstate = my_trainer.fit(dataloader_training=trainloader,
                                    epochs=argv.epochs,
                                    validation_set=mnist_data_test)

#  def register_nan_checks_(model):
#        def check_grad(module, input, output):
#            if not hasattr(module, "weight"):
#                return
#            if module.weight is None or module.weight.grad is None:
#                return
#           # if (module.weight.grad.abs() == 0).any():
#           #     print('Gradient in ' + type(module).__name__)
#           #     print(module.weight.grad)
#           #     print(module.extra_repr)
#            #if (module.weight.grad.abs() > 1.).any():
#            #    print('Gradient in ' + type(module).__name__)
#            #    print(module.weight.grad)
#            #    print(module.extra_repr)
#            if (module.weight.grad != module.weight.grad).any():
#                print('NaN Gradients in ' + type(module).__name__)
#                print(module.weight.grad)
#                print(module.extra_repr)
#            if module.weight.grad.abs().max() > 10000.:
#                print('Exploding Gradients in ' + type(module).__name__)
#                print(module.weight.grad)
#                print(module.extra_repr)
#        handles = []
#        for module in model.modules():
#            handles.append(module.register_forward_hook(check_grad))
#        return handles
#
#    register_nan_checks_(gmvae)

     
