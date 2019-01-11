print("\n##################################################")
print("LAST TESTED SYSTEM ENVIRONMENT")
print("0.4 (torch.__version__)")
print("3.5 (ummon.version)")
print("##################################################\n")

      
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ummon import *


class ModelNormal(nn.Module):
    
    def __init__(self, input_features = 784, hidden_layer = 50, latent_features = 50):
        super(ModelNormal, self).__init__()   
        
        self.affine_enc1 = nn.Linear(input_features , hidden_layer, bias=False)
        self.affine_m = nn.Linear(hidden_layer, latent_features, bias=False)
        self.affine_v = nn.Linear(hidden_layer, latent_features, bias=False)
        self.affine_r = nn.Linear(hidden_layer, 1, bias=False)
        self.affine_dec = nn.Linear(latent_features, input_features)
        self.dropout_dec = nn.Dropout(p=0.5)
        
        self.m = None
        self.v = None
        self.z = None
        self.o = None
        self.r = None

    def forward(self, x):
        h1 = F.relu(self.affine_enc1(x))

        m = self.affine_m(h1)
        v = F.softplus(self.affine_v(h1))
        
        dist_normal = torch.distributions.Normal(0 * m, 1)
        e = dist_normal.sample()
        z = (e * v) + m
        
        r = self.affine_r(h1)        
        o = F.sigmoid(self.affine_dec(z))
        
        self.r = r
        self.z = z
        self.v = v.detach()
        self.m = m.detach()
        self.o = o
        return self.z
    
class ModelLogNormal(ModelNormal):
    
    def __init__(self, *args, **kwargs):
        super(ModelLogNormal, self).__init__(*args, **kwargs)   

    def forward(self, x):
        h1 = F.relu(self.affine_enc1(x))

        m = self.affine_m(h1)
        v = F.softplus(self.affine_v(h1))
        
        dist_log_normal = torch.distributions.LogNormal(m, v)
        z = dist_log_normal.rsample()
        
        r = self.affine_r(h1)        
        o = F.sigmoid(self.affine_dec(z))
        
        self.r = r
        self.z = z
        self.v = v.detach()
        self.m = m.detach()
        self.o = o
        return self.z
    
class ModelLaplace(ModelNormal):
    
    def __init__(self, *args, **kwargs):
        super(ModelLaplace, self).__init__(*args, **kwargs)   

    def forward(self, x):
        h1 = F.relu(self.affine_enc1(x))

        m = self.affine_m(h1)
        v = F.softplus(self.affine_v(h1))
        
        dist_laplace = torch.distributions.Laplace(m, v)
        z = dist_laplace.rsample()
        
        r = self.affine_r(h1)        
        o = F.sigmoid(self.affine_dec(z))
        
        self.r = r
        self.z = z
        self.v = v.detach()
        self.m = m.detach()
        self.o = o
        return self.z
 

class AffineModel(nn.Module):
    
    def __init__(self, input_features=768, hidden_features=20, output_features = 1):
        super(AffineModel, self).__init__()   
        
        self.affine1 = nn.Linear(input_features , hidden_features, bias=False)
        self.affine2 = nn.Linear(hidden_features, output_features, bias=False)
    
    def forward(self, x):
        h1 = F.relu(self.affine1(x))
        m = self.affine2(h1)
        return m
    
class ModelLaplaceWithSimilarity(ModelLaplace):
    
    def __init__(self, gan_hidden_features = 20, *args, **kwargs):
        super(ModelLaplaceWithSimilarity, self).__init__(*args,** kwargs)   
        
        self.similarity = AffineModel(kwargs["input_features"], gan_hidden_features, gan_hidden_features)
    
class ModelLaplaceWithGan(ModelLaplace):
    
    def __init__(self, gan_hidden_features = 20, *args, **kwargs):
        super(ModelLaplaceWithGan, self).__init__(*args,** kwargs)   
        
        self.descriminator = AffineModel(kwargs["input_features"], gan_hidden_features, 1)
       