import argparse
import sys
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from ummon import *
from negvarbound import *
from similarityloss import *
from model import *
from helpers import Evaluator
import helpers

parser = argparse.ArgumentParser(description='PyTorch Novelty Detection')

# TRAINING PARAMS
parser.add_argument('--epochs', type=int, default=100, metavar='',
                    help='Amount of epochs for training (default: 100)')
parser.add_argument('--batch_size', type=int, default=1000, metavar='',
                    help='Batch size for SGD (default: 100)')
parser.add_argument('--lrate', type=float, default=0.0001, metavar="",
                    help="Learning rate (default: 0.001")
parser.add_argument('--with_cuda', action='store_true', dest='use_cuda',
                    help="Shall cuda be used (default: False)")
parser.add_argument('--plots', action='store_true', dest='plots',
                    help="Shall matplotlib be used (default: False)")

argv = parser.parse_args()
sys.argv = [sys.argv[0]] 

torch.manual_seed(4)


if __name__ == '__main__':
    
    transform = transforms.Compose(
        [transforms.ToTensor(), helpers.flatten_transform])

    patches_reference = ImagePatches("/ext/data/cut.png", mode='gray', train_percentage=1.0,
                                     train=True, stride_y=16,
                                     stride_x=16, window_size=32, transform=transform)

    patches_noise = ImagePatches("/ext/data/cut_noise.png", mode='gray', train_percentage=1.0,
                                 train=True, stride_y=16,
                                 stride_x=16, window_size=32, transform=transform)

    patches_anaomaly = AnomalyImagePatches("/ext/data/cut.png", mode='gray', train=False,
                                           stride_y=16,
                                           stride_x=16,
                                           window_size=32, transform=transform, train_percentage=0, propability=1.0,
                                           anomaly=SquareAnomaly(size=8, color=0))

    # Train ref
    data_train_ref = [patches_reference[i][0].data for i in range(len(patches_reference))]
    data_train_ref = torch.stack(data_train_ref).numpy() / 100

    # Train noise
    data_train_noise = [patches_noise[i][0].data for i in range(len(patches_noise))]
    data_train_noise = torch.stack(data_train_noise).numpy() / 100

    # Train
    data_train = np.array((data_train_ref, data_train_noise))

    # Novelty
    data_novelty = [patches_anaomaly[i][0].data for i in range(len(patches_anaomaly))]
    data_novelty = torch.stack(data_novelty).numpy() / 100


    
    
######################################################
# NORMAL DISTRIBUTION
######################################################
    
    # Model    
    model = ModelNormal(input_features = data_train_ref.shape[1], hidden_layer=250, latent_features=50)
    
    # LOSS        
    criterion = KLLoss(model=model, size_average=False)

    # INSTANTIATE OPTIMIZER
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=1)
    
    
    # Activate matplotlib
    argv.plots = True
    
    with Logger(loglevel=10, log_batch_interval=1000) as lg:
            
            
        # CREATE A TRAINER
        my_trainer = UnsupervisedTrainer(lg, 
                            model, 
                            criterion, 
                            optimizer, 
                            trainingstate = Trainingstate(),
                            model_filename="KL_MIN",
                            use_cuda= argv.use_cuda,
                            profile = False,
                            convergence_eps = 1e-5)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=(data_train_ref, 20),
                                    epochs=100)
    
    #Evaluator
    evaluator = Evaluator(model, data_train_ref, data_train_noise, data_novelty)
    evaluator.evaluate_model(argv)
    
######################################################
# LAPLACE
######################################################

    # Model    
    model = ModelLaplace(input_features = data_train_ref.shape[1], hidden_layer=250, latent_features=25)
    
    torch.manual_seed(4)
    
    # LOSS        
    criterion = KLLoss_laplace(model=model, size_average=False, mean=2, scale=0.5)
    
    # INSTANTIATE OPTIMIZER
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.1)
    
    
    # Activate matplotlib
    argv.plots = True
    
    with Logger(loglevel=10, log_batch_interval=1000) as lg:
        
        # TRAININGSTATE
        ts = Trainingstate()
        
            
        # CREATE A TRAINER
        my_trainer = UnsupervisedTrainer(lg, 
                            model, 
                            criterion, 
                            optimizer, 
                            trainingstate = ts,
                            model_filename="KL_MIN",
                            use_cuda= argv.use_cuda,
                            profile = False,
                            convergence_eps = 1e-2)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=(data_train_ref, 20),
                                    epochs=100)
    
    #Evaluator
    evaluator = Evaluator(model, data_train_ref, data_train_noise, data_novelty)
    evaluator.evaluate_model(argv)
    
