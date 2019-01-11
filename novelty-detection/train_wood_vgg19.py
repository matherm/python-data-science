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
parser.add_argument('--model', type=int, default=0,
                    help="Which model to train (0=KLminimizer, 1=Euclidean-Minimizer) (default: 0)")
parser.add_argument('--plots', action='store_true', dest='plots',
                    help="Shall matplotlib be used (default: False)")
parser.add_argument('--grid', action='store_true', dest='grid',
                    help="Grid search (default: False)")

argv = parser.parse_args()
sys.argv = [sys.argv[0]] 

from ummon import *
from negvarbound import *
from model import *
from helpers import Evaluator
import helpers

torch.manual_seed(4)

if __name__ == '__main__':
    
    # WOOD
    transform = transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), VGG19Features("pool4"), helpers.flatten_transform])
    wood_data = ImagePatches("/ext/data/Wood-0035.png", mode='rgb', train=True, stride_y=14, stride_x=14, window_size=28, transform=transform)
    wood_data_test = AnomalyImagePatches("/ext/data/Wood-0035.png", mode='rgb', train=True, stride_y=14, stride_x=14, window_size=28, transform=transform, propability=1.0, anomaly=SquareAnomaly(size=8, color=255))
    
    wood_data = [wood_data[i][0].data for i in range(len(wood_data))]
    wood_data = torch.stack(wood_data).numpy() / 10
    
    wood_data_test  = [wood_data_test[i][0].data for i in range(len(wood_data_test))]
    wood_data_test  = torch.stack(wood_data_test).numpy() / 10
    
    
    # Novelty
    data_novelty = wood_data_test
    
    # Train
    data_train = wood_data
    
    # Val
    data_val = data_train
    
    
######################################################
# NORMAL DISTRIBUTION
######################################################
    
    # Model    
    model = ModelNormal(input_features = data_train.shape[1], hidden_layer=20, latent_features=20)
    
    torch.manual_seed(4)

    # LOSS        
    criterion = KLLoss(model=model, size_average=False)

    # INSTANTIATE OPTIMIZER
    optimizer = torch.optim.SGD(model.parameters(), lr=argv.lrate, weight_decay=1)
    
    #Evaluator
    evaluator = Evaluator(model, data_train, data_val, data_novelty)
    
    # Activate matplotlib
    argv.plots = True
    
    with Logger(loglevel=10, log_batch_interval=601) as lg:
            
            
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
        my_trainer.fit(dataloader_training=(wood_data, 20),
                                    epochs=200)
    
    evaluator.evaluate_model(argv)
    
    
######################################################
# LOGNORMAL
######################################################
    
    # Model    
    model = ModelLogNormal(input_features = data_train.shape[1], hidden_layer=20, latent_features=20)
    
    torch.manual_seed(4)
    
    # LOSS        
    criterion = KLLoss_lognormal(model=model, size_average=False)
    
    # INSTANTIATE OPTIMIZER
    optimizer = torch.optim.SGD(model.parameters(), lr=argv.lrate, weight_decay=1)
    
    #Evaluator
    evaluator = Evaluator(model, data_train, data_val, data_novelty)
    
    # Activate matplotlib
    argv.plots = True
    
    with Logger(loglevel=10, log_batch_interval=601) as lg:
            
            
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
        my_trainer.fit(dataloader_training=(data_train, 20),
                                    epochs=argv.epochs)
    
    evaluator.evaluate_model(argv)


######################################################
# LAPLACE
######################################################

    # Model    
    model = ModelLaplace(input_features = data_train.shape[1], hidden_layer=20, latent_features=20)
    
    torch.manual_seed(4)
    
    # LOSS        
    criterion = KLLoss_laplace(model=model, size_average=False, mean=2, scale=0.5)
    
    # INSTANTIATE OPTIMIZER
    optimizer = torch.optim.SGD(model.parameters(), lr=0.000001, weight_decay=1)
    
    #Evaluator
    evaluator = Evaluator(model, data_train, data_val, data_novelty)
    
    # Activate matplotlib
    argv.plots = True
    
    with Logger(loglevel=10, log_batch_interval=601) as lg:
            
            
        # CREATE A TRAINER
        my_trainer = UnsupervisedTrainer(lg, 
                            model, 
                            criterion, 
                            optimizer, 
                            trainingstate = Trainingstate(),
                            model_filename="KL_MIN",
                            use_cuda= argv.use_cuda,
                            profile = False,
                            convergence_eps = 1e-1)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=(data_train, 20),
                                    epochs=300)
    
    evaluator.evaluate_model(argv)
    
    # {'AUROC LAT (TRAIN)': 0.8743801652892562,
    # 'AUROC LAT (VAL)': 0.8661157024793389,
    # 'AUROC REC (TRAIN)': 0.86900826446281,
    # 'AUROC REC (VAL)': 0.8528925619834712}
    
######################################################
# LAPLACE WITH R-SHIFT
######################################################


    class CombinedLoss(nn.Module):
        
        def __init__(self, model, *args, **kwargs):
            super(CombinedLoss, self).__init__()   
            self.model = model
            self.r_shift = KLLoss_shift_r(model=model, size_average=False)
            self.kl_loss = KLLoss_laplace(model=model, size_average=False, mean=10, scale=0.3)
        
        def forward(self, inpt, outpt):
            self.r_shift()
            return self.kl_loss(inpt,outpt)


    # Model    
    model = ModelLaplace(input_features = data_train.shape[1], hidden_layer=20, latent_features=20)
    
    torch.manual_seed(4)
    
    # LOSS        
    criterion = CombinedLoss(model)
    
    # INSTANTIATE OPTIMIZER
    optimizer = torch.optim.SGD(model.parameters(), lr=argv.lrate, weight_decay=1)
    
    #Evaluator
    evaluator = Evaluator(model, data_train, data_val, data_novelty)
    
    # Activate matplotlib
    argv.plots = True
    
    with Logger(loglevel=10, log_batch_interval=601) as lg:
            
            
        # CREATE A TRAINER
        my_trainer = UnsupervisedTrainer(lg, 
                            model, 
                            criterion, 
                            optimizer, 
                            trainingstate = Trainingstate(),
                            model_filename="KL_MIN",
                            use_cuda= argv.use_cuda,
                            profile = False,
                            convergence_eps = 1e-3)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=(data_train, 20),
                                    epochs=200)
    
    evaluator.evaluate_model(argv)
    
    
    # {'AUROC LAT (TRAIN)': 0.8590909090909091,
    # 'AUROC LAT (VAL)': 0.8752066115702479,
    # 'AUROC REC (TRAIN)': 0.8677685950413224,
    # 'AUROC REC (VAL)': 0.8619834710743801}
    
    
