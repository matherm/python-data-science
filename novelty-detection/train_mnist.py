import argparse
import sys
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
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
import helpers
from helpers import Evaluator

torch.manual_seed(4)
TRAINING_SET = 4


if __name__ == '__main__':
    
    # MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_data = MNIST("/ext/data/mnist/", train=True, transform=transform, target_transform=None, download=True)
    mnist_data_test = MNIST("/ext/data/mnist/", train=False, transform=transform, target_transform=None, download=True)

    argv.epochs = 1000

    #
    # TEST
    #
    data_test, labels_test = zip(*[(d[0],d[1].item()) for _, d in zip(range(1000), mnist_data_test)])
    data_test = torch.cat(data_test).reshape(1000,-1).numpy()
    labels_test = np.asarray(labels_test, dtype=np.float32)
    
    # Subset
    labels_val = labels_test[labels_test == TRAINING_SET]
    data_val = data_test[labels_test == TRAINING_SET]
    
    #
    # TRAINING
    #
    data, labels = zip(*[(d[0],d[1].item()) for _, d in zip(range(60000), mnist_data)])
    data = torch.cat(data).reshape(60000,-1).numpy()
    labels = np.asarray(labels, dtype=np.float32)
    
    # Subset
    labels_train = labels[labels == TRAINING_SET][0:220]
    data_train = data[labels == TRAINING_SET][0:220]

    # Novelty
    data_novelty = np.vstack((data_test[labels_test == 0][0:5],
                              data_test[labels_test == 7][0:3],
                              data_test[labels_test == 9][0:3]))
    
######################################################
# LAPLACE WITH GAN LOSS
######################################################
    torch.manual_seed(4)
    
    def debug_backward(output, targets, loss):
        print(model.descriminator.affine2.weight.grad.sum())

    # Model    
    model = ModelLaplaceWithGan(input_features = data_train.shape[1], hidden_layer=20, latent_features=20)
    
    # INSTANTIATE OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters())#, lr=0.0001)
    
    # LOSS        
    criterion = KLLoss_laplace_with_gan_loss(model=model, size_average=False, mean=10, scale=1)
    
    # Activate matplotlib
    argv.plots = True
    
    with Logger(loglevel=20, log_batch_interval=601) as lg:
            
            
        # CREATE A TRAINER
        my_trainer = UnsupervisedTrainer(lg, 
                            model, 
                            criterion, 
                            optimizer, 
                            trainingstate = Trainingstate(),
                            model_filename="/dev/null/KL_MIN",
                            use_cuda= argv.use_cuda,
#                            after_backward_hook=debug_backward,
                            profile = False)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=(data_train, 20),
                                    epochs=argv.epochs)
        
        
        #Evaluator
        evaluator = Evaluator(model, data_train, data_val, data_novelty)
        evaluator.evaluate_model(argv)
    
        print = lg.info
        helpers.compareOneClassSVM(0.001, data_train, data_novelty, kernel='linear', print=print)
        helpers.compareOneClassSVM(0.001, data_train, data_novelty, kernel='rbf',    print=print)
######################################################
# LAPLACE WITH SIMILARITY LOSS
######################################################
    torch.manual_seed(4)

    # Model    
    model = ModelLaplaceWithSimilarity(input_features = data_train.shape[1], hidden_layer=20, latent_features=20)
        
    # LOSS        
    criterion = KLLoss_laplace_with_similarity_loss(model=model, train_similarity_model=True, size_average=False, mean=10, scale=1)
    
    # INSTANTIATE OPTIMIZER
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    
    # Activate matplotlib
    argv.plots = True
    
    with Logger(loglevel=20, log_batch_interval=601) as lg:
            
            
        # CREATE A TRAINER
        my_trainer = UnsupervisedTrainer(lg, 
                            model, 
                            criterion, 
                            optimizer, 
                            trainingstate = Trainingstate(),
                            model_filename="/dev/null/KL_MIN",
                            use_cuda= argv.use_cuda,
                            profile = False)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=(data_train, 20),
                                    epochs=argv.epochs)
        
        
        #Evaluator
        evaluator = Evaluator(model, data_train, data_val, data_novelty)
        evaluator.evaluate_model(argv)
    
        print = lg.info
        helpers.compareOneClassSVM(0.001, data_train, data_novelty, kernel='linear', print=print)
        helpers.compareOneClassSVM(0.001, data_train, data_novelty, kernel='rbf',  print=print)

    
    # AUROC LAT (VAL) 0.9305785123966942
    # AUROC REC (VAL) 0.9338842975206612
    # AUROC LAT (TRAIN) 0.9309917355371901
    # AUROC REC (TRAIN) 0.9247933884297521
  
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
                            model_filename="/dev/null/KL_MIN",
                            use_cuda= argv.use_cuda,
                            profile = False)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=(data_train, 20),
                                    epochs=argv.epochs)
    
    evaluator.evaluate_model(argv)
    
    # AUROC LAT (VAL) 0.8710743801652893
    # AUROC REC (VAL) 0.8801652892561984
    # AUROC LAT (TRAIN) 0.8776859504132231
    # AUROC REC (TRAIN) 0.8983471074380165
    
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
                            profile = False)
        
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
    criterion = KLLoss_laplace(model=model, size_average=False, mean=10, scale=1)
    
    # INSTANTIATE OPTIMIZER
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    
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
                            profile = False)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=(data_train, 20),
                                    epochs=argv.epochs)
    
    evaluator.evaluate_model(argv)
    
    # AUROC LAT (VAL) 0.9305785123966942
    # AUROC REC (VAL) 0.9338842975206612
    # AUROC LAT (TRAIN) 0.9309917355371901
    # AUROC REC (TRAIN) 0.9247933884297521
    
######################################################
# LAPLACE WITH R-SHIFT
######################################################


    class CombinedLoss(nn.Module):
        
        def __init__(self, model, *args, **kwargs):
            super(CombinedLoss, self).__init__()   
            self.model = model
            self.r_shift = KLLoss_shift_r(model=model, size_average=False)
            self.kl_loss = KLLoss_laplace(model=model, size_average=False, mean=0, scale=1)
        
        def forward(self, inpt, outpt):
            self.r_shift()
            return self.kl_loss(inpt,outpt)


    # Model    
    model = ModelLaplace(input_features = data_train.shape[1], hidden_layer=20, latent_features=10)
    
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
                            profile = False)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=(data_train, 20),
                                    epochs=argv.epochs)
    
    evaluator.evaluate_model(argv)
    
    
    # {'AUROC LAT (TRAIN)': 0.8590909090909091,
    # 'AUROC LAT (VAL)': 0.8752066115702479,
    # 'AUROC REC (TRAIN)': 0.8677685950413224,
    # 'AUROC REC (VAL)': 0.8619834710743801}
    

