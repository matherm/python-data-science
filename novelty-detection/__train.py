import argparse
import sys
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import pickle
def save_obj(obj, name ):
    with open('local/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('local/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def flatten_transform(tensor):
    return tensor.view(-1)   


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
import ummon.utils as uu
from ummon.unsupervised import UnsupervisedTrainer
from ummon.logger import Logger
from ummon.datasets.curet import *
from ummon.trainingstate import Trainingstate
from model import KLLoss
from model import Model

torch.manual_seed(4)

TRAINING_SET = 4
DATA = "MNIST"

def predict_sample_data():
    noise_u = np.random.uniform(0,1,data_val.shape[1]*data_val.shape[0]).reshape(-1, data_val.shape[1]).astype(np.float32)
    noise_w = np.random.normal(0,1,data_val.shape[1]*data_val.shape[0]).reshape(-1, data_val.shape[1]).astype(np.float32)

    noise_u_t = Predictor.predict(model, noise_u)
    noise_w_t = Predictor.predict(model, noise_w)
    val_t = Predictor.predict(model, data_val)
    val_o = model.o.data.numpy()
    novel_t = Predictor.predict(model, data_novelty)
    novel_o = model.o.data.numpy()
    train_t = Predictor.predict(model, data_train)
    train_o = model.o.data.numpy()
    return noise_u_t, noise_w_t, val_t, val_o, novel_t, novel_o, train_t, train_o


def evaluate_model(*args):
    
    noise_u_t, noise_w_t, val_t, val_o, novel_t, novel_o, train_t, train_o = predict_sample_data()
    
    if argv.plots == True:
        import matplotlib.pyplot as plt
        clip = 50000
        plt.hist(np.clip(np.sum(train_t**2, 1), 0, clip), color="b")
        plt.hist(np.clip(np.sum(noise_u_t**2, 1), 0, clip), color="g")
        plt.hist(np.clip(np.sum(noise_w_t**2, 1), 0, clip), color="y")
        plt.hist(np.clip(np.sum(val_t**2, 1), 0, clip))
        plt.hist(np.clip(np.sum(novel_t**2, 1), 0, clip), color="r")
        plt.title("Distance to origin (euclidean) ")
        plt.show()
        plt.hist(np.clip(np.sum((train_o - data_train)**2, 1), 0, clip), color="b")
        plt.hist(np.clip(np.sum((val_o - data_val)**2, 1), 0, clip))
        plt.hist(np.clip(np.sum((novel_o -data_novelty)**2, 1), 0, clip), color="r") 
        plt.title("MSE (reconstruction loss)")
        plt.show()
        plt.hist(np.sum(train_t, 1), color="b")
        plt.hist(np.sum(noise_u_t, 1), color="g")
        plt.hist(np.sum(noise_w_t, 1), color="y")
        plt.hist(np.sum(val_t, 1))
        plt.hist(np.sum(novel_t, 1), color="r")
        plt.title("Sum of means (isotropic gaussian)")
        plt.show()
        try:
            rect = int(data_val.shape[1]**0.5)
            plt.imshow(model.o[10].data.numpy().reshape(rect,rect))
            plt.title("Reconstruction (good)")
            plt.show()
            plt.imshow(model.affine_dec(torch.from_numpy(np.random.uniform(-10,10, noise_u_t.shape[1])).float()).data.numpy().reshape(rect,rect))
            plt.title("Reconstruction (unform random)")
            plt.show()
            plt.imshow(novel_o[0].reshape(rect,rect))
            plt.title("Reconstruction (novelty - zero)")
            plt.show()
            plt.imshow(novel_o[10].reshape(rect,rect))
            plt.title("Reconstruction (novelty - nine)")
            plt.show()
            plt.imshow(novel_o[6].reshape(rect,rect))
            plt.title("Reconstruction (novelty - seven)")
            plt.show()
        except:
            pass

    from sklearn.metrics import roc_auc_score
    arocs = {}
    y_true = np.array([1] * val_t.shape[0] + [-1] * novel_t.shape[0])
    y_scores = np.hstack((np.sum(val_t**2, 1), np.sum(novel_t**2, 1)))
    print("AUROC LAT (VAL)", roc_auc_score(y_true, y_scores))
    arocs["AUROC LAT (VAL)"] = roc_auc_score(y_true, y_scores)
    
    y_true = np.array([-1] * val_t.shape[0] + [1] * novel_t.shape[0])
    y_scores = np.hstack((np.sum((data_val - val_o)**2, 1), np.sum((novel_o-data_novelty)**2, 1)))
    print("AUROC REC (VAL)", roc_auc_score(y_true, y_scores))
    arocs["AUROC REC (VAL)"] = roc_auc_score(y_true, y_scores)
    
    
    y_true = np.array([1] * train_t.shape[0] + [-1] * novel_t.shape[0])
    y_scores = np.hstack((np.sum(train_t**2, 1), np.sum(novel_t**2, 1)))
    print("AUROC LAT (TRAIN)", roc_auc_score(y_true, y_scores))
    arocs["AUROC LAT (TRAIN)"] = roc_auc_score(y_true, y_scores)
    
    
    y_true = np.array([-1] * train_t.shape[0] + [1] * novel_t.shape[0])
    y_scores = np.hstack((np.sum((data_train - train_o)**2, 1), np.sum((novel_o-data_novelty)**2, 1)))
    print("AUROC REC (TRAIN)", roc_auc_score(y_true, y_scores))
    arocs["AUROC REC (TRAIN)"] = roc_auc_score(y_true, y_scores)
    return arocs


if __name__ == '__main__':
    
    if DATA == "AE":
        mnist_data_cae = np.load("/ext/data/one-class-nn-data/our_train_features.npz")["arr_0"]
        mnist_data_cae_test = np.load("/ext/data/one-class-nn-data/our_test_features.npz")["arr_0"]
        
        data_val = mnist_data_cae / 1000
        #data_val = (mnist_data_cae - mnist_data_cae.mean()) / mnist_data_cae.std()
        labels_val = np.zeros(data_val.shape[0]) + 4
        #data_train = (mnist_data_cae - mnist_data_cae.mean()) / mnist_data_cae.std() 
        data_train = mnist_data_cae / 1000
        labels = np.zeros(data_val.shape[0]) + 4

        # Novelty
        data_novelty = (mnist_data_cae_test) / 1000
        
        # Model    
        model = Model(input_features = data_train.shape[1], hidden_layer=20, latent_features=20, 
                      with_non_gauss=False,
                      learn_r=False, 
                      learn_rec=True, 
                      learn_assym = True,
                      shift_r = False,
                      lognormal = False,
                      scale = 1,
                      mean = 0)   
        
    elif DATA == "CURET":
        curet_data = CuretVGG19grams(path="/ext/data/curet_vgg19_grams/", download=True, features="pool4")

    elif DATA == "CIFAR":
        
        transform = transforms.Compose([transforms.ToTensor(),VGG19Features()])
        cifar_data = CIFAR10("/ext/data/cifar10/", train=True, transform=transform, target_transform=None, download=True)
        cifar_data_test = CIFAR10("/ext/data/cifar10/", train=False, transform=transform, target_transform=None, download=True)
        
        #
        # TEST
        #
        data_test, labels_test = zip(*[(d[0],d[1]) for _, d in zip(range(1000), cifar_data_test)])
        data_test = torch.cat(data_test).reshape(1000,-1).numpy()
        labels_test = np.asarray(labels_test, dtype=np.int32)
        
        # Subset
        labels_val = labels_test[labels_test == TRAINING_SET]
        data_val = data_test[labels_test == TRAINING_SET]
        
        #
        # TRAINING
        #
        data, labels = zip(*[(d[0],d[1]) for _, d in zip(range(50000), cifar_data)])
        data = torch.cat(data).reshape(50000,-1).numpy()
        labels = np.asarray(labels, dtype=np.float32)
        
        # Subset
        labels_train = labels[labels == TRAINING_SET][0:220]
        data_train = data[labels == TRAINING_SET][0:220]
    
        # Novelty
        data_novelty = np.vstack((data_test[labels_test == 0][0:5],
                                  data_test[labels_test == 7][0:3],
                                  data_test[labels_test == 9][0:3]))
        
        # Model    
        model = Model(input_features = data_train.shape[1], hidden_layer=20, latent_features=1, 
                      with_non_gauss=False,
                      learn_r=False, 
                      learn_rec=True, 
                      learn_assym = False,
                      shift_r = False,
                      lognormal = False,
                      scale = 0.1,
                      mean = 100)   
    
    else:
        if DATA == "MNIST":
            # MNIST
            transform = transforms.Compose([transforms.ToTensor()])
            mnist_data = MNIST("/ext/data/mnist/", train=True, transform=transform, target_transform=None, download=True)
            mnist_data_test = MNIST("/ext/data/mnist/", train=False, transform=transform, target_transform=None, download=True)
        
            argv.epochs = 100
        
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
    
            # Model    
            model = Model(input_features = data_train.shape[1], hidden_layer=10, latent_features=2, 
                          with_non_gauss=False,
                          learn_r=False, 
                          learn_rec=True, 
                          learn_assym = False,
                          shift_r = False,
                          lognormal = True,
                          scale = 1,
                          mean = 0)   
       
    if not argv.grid:
        torch.manual_seed(4)
        argv.plots = True

        # LOSS        
        criterion = KLLoss(model, size_average=False)
    
        # INSTANTIATE OPTIMIZER
        optimizer = torch.optim.SGD(model.parameters(), lr=argv.lrate, weight_decay=1)
        
        with Logger(loglevel=10, log_batch_interval=601) as lg:
                
                
            # CREATE A TRAINER
            my_trainer = UnsupervisedTrainer(lg, 
                                model, 
                                criterion, 
                                optimizer, 
                                trainingstate = Trainingstate(),
                                model_filename="KL_MIN",
                                use_cuda= argv.use_cuda,
                              #  after_epoch_hook = evaluate_model,
                                profile = False)
            
            # START TRAINING
            my_trainer.fit(dataloader_training=(data_train, 20),
                                        epochs=argv.epochs)
        
            evaluate_model()
    
    else:
        result = []
        for d in range(220,6000,500):
            # Subset
            labels_train = labels[labels == TRAINING_SET][0:d]
            data_train = data[labels == TRAINING_SET][0:d]
            for lognormal in [True, False]:
                for hidden in range(5, 30, 5):
                    for latent in range(2, 20, 2):
                        for with_r in [True, False]:
                            for with_recon in [True, False]:
                                for with_non_gauss in [True, False]:
                                    for assym in [True, False]:
                                        try:
                                            # SEED IT
                                            torch.manual_seed(4)
                                            
                                            # INTANTIATE LOSS
                                            model = Model(input_features = data_train.shape[1], hidden_layer=10, latent_features=3, 
                                                          with_non_gauss=with_non_gauss,
                                                          learn_r=False, 
                                                          learn_rec=with_recon, 
                                                          learn_assym = assym,
                                                          shift_r = with_r,
                                                          lognormal = lognormal,
                                                          scale = 1,
                                                          mean = 0)   
    
                                            criterion = KLLoss(model, size_average=False)
                                            
                                            # INSTANTIATE OPTIMIZER
                                            optimizer = torch.optim.SGD(model.parameters(), lr=argv.lrate, weight_decay=1)
                                            
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
                                                
                                                
                                                arocs = evaluate_model()
                                        
                                        
                                            noise_u_t, noise_w_t, val_t, val_o, novel_t, novel_o, train_t, train_o = predict_sample_data()
                                            
                                            result.append((d, lognormal, hidden, 
                                                           latent, with_r, with_recon, with_non_gauss, assym,
                                                           noise_u_t, noise_w_t, val_t, novel_t, 
                                                           arocs))
                                        
                                            save_obj(result, "state.pkl")
                                       
                                            import os
                                            [os.remove(f) for f in os.listdir("./") if f.endswith(".pth.tar")]
                                        except:
                                            print("FAILED")
                                            pass
  