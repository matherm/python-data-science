# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Real Gaussian Mixture VAE

[1] Rui Shu 2016.    
    http://ruishu.io/2016/12/25/gmvae/
    
Generative Model
----------------
p(x,y,z) = p(y)*p(z|y)*p(x|z)

y ~ Cat(1/K)
z ~ N(mu(y), var(y))
x ~ B(mu(z))
    
Inference Model
---------------
q(y,z|x) = q(y|x)*q(z|x,y)


Loss (Variational Lower-Bound)
----
L = Exp_q(y,z|x) [ln p(x,y,z) - ln q(y,z|x)]
  = Exp_q(y,z|x) [ln  p(y)  + p(z|y)   +  ln(p(x|y,z)]
                     -----    -----
                     q(y|x)   q(z|x,y)
                     
                     
Created on Mon Jan 15 13:01:27 2018

@author: matthias
@copyright: IOS
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ummon import *

print("\n##################################################")
print("LAST TESTED SYSTEM ENVIRONMENT")
print("0.4 (torch.__version__)")
print("3.5 (ummon.version)")
print("##################################################\n")

class P_x(nn.Module):
    
    def __init__(self, input_features = 784, latent_features = 64):
        super(P_x, self).__init__()   
        self.affine_dec1 = nn.Linear(latent_features, 512, bias=False)
        self.affine_dec2 = nn.Linear(512, 512, bias=False)
        self.affine_dec3 = nn.Linear(512, input_features, bias=False)

    def forward(self, z):
        h1 = F.relu(self.affine_dec1(z))
        h2 = F.relu(self.affine_dec2(h1))
        px = self.affine_dec3(h2)
        return px
    
class P_z(nn.Module):
    
    def __init__(self, latent_features = 64, mixtures_count = 10):
        super(P_z, self).__init__()   
        self.means = nn.Linear(mixtures_count, latent_features, bias=False)
        self.vars  = nn.Linear(mixtures_count, latent_features, bias=False)

    def forward(self, y):
        zv = F.softplus(self.vars(y))
        zm = self.means(y)
        return zm, zv
    
class Q_y(nn.Module):
    
    def __init__(self, input_features = 784, mixtures_count = 10):
        super(Q_y, self).__init__()   
        self.affine_prop1 = nn.Linear(input_features, 512, bias=False)
        self.affine_prop2 = nn.Linear(512, 512, bias=False)
        self.affine_prop3 = nn.Linear(512, mixtures_count)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.affine_prop1(x))
        h2 = F.relu(self.affine_prop2(h1))
        qy_linear = self.affine_prop3(h2)
        qy = F.softmax(qy_linear, dim=1)
        return qy, qy_linear
    
class Q_z(nn.Module):
    
    def __init__(self, input_features = 784, latent_features = 64, mixtures_count = 10):
        super(Q_z, self).__init__()   
        self.affine_enc1 = nn.Linear(input_features + mixtures_count, 512, bias=False)
        self.affine_enc2 = nn.Linear(512, 512, bias=False)
        self.affine_enc3 = nn.Linear(512, latent_features)
        self.affine_enc4 = nn.Linear(512, latent_features)

    def forward(self, x, y):
        x = x.view(x.size(0), -1)
        xy = torch.cat((x, y), dim=1)
        h1 = F.relu(self.affine_enc1(xy))
        h2 = F.relu(self.affine_enc2(h1))
        
        zm = self.affine_enc3(h2)
        zv = F.softplus(self.affine_enc4(h2))
        m = torch.distributions.Normal(0 * zm, 1)
        e = m.sample()
        z = (e * zv) + zm
        return z, zm.detach(), zv.detach()
    

class GMVAE(nn.Module):
    
    def __init__(self, input_features = 784, latent_features = 64, mixtures_count = 20):
        super(GMVAE, self).__init__()    
        self.is_cuda = False
        self.mixtures_count = mixtures_count
    
        # Priors
        self.qy = Q_y(input_features=input_features, mixtures_count=mixtures_count)
        self.pz = P_z(latent_features=latent_features, mixtures_count=mixtures_count)

        # Encoder  
        self.qz = Q_z(input_features=input_features, latent_features=latent_features, mixtures_count=mixtures_count)
        
        # Decoder
        self.px = P_x(input_features=input_features, latent_features=latent_features)
        
        # Inference model
        self.z = None
        self.zm = None
        self.zv = None 
        self.zm_prior = None
        self.zv_prior = None
        
        # A-priori cluster assignment vector cache
        self.y = None
        
    def cuda(self):
        self.is_cuda = True
        return super(GMVAE, self).cuda()
        
    def cpu(self):
        self.is_cuda = False
        return super(GMVAE, self).cpu()
        
    def forward(self, x):
         # Create fresh cluster assignment vectors
        if self.y is None or x.size(0) != self.y.size(0):
             mx = self.mixtures_count
             y_numpy = np.repeat(np.eye(mx).reshape(1,mx,mx), x.size(0), axis=0).astype(x.cpu().data.numpy().dtype)
             self.y = Variable(torch.from_numpy(y_numpy))
             if self.is_cuda:
                 self.y = self.y.cuda()
        
        # for each proposed y, infer z and reconstruct x
        y = self.y
        z, zm, zv, zm_prior, zv_prior, x_recon = [[None] * self.mixtures_count for i in range(6)]
        for i in range(self.mixtures_count):
            zm_prior[i], zv_prior[i] = self.pz(y[:,i])       # [B x LatentDims,..,mixtures_count]
            z[i], zm[i], zv[i] = self.qz(x, y[:,i])          # [B x LatentDims,..,mixtures_count]
            x_recon[i] = self.px(z[i])                       # [B x 768,..,mixtures_count]

        # Remember Inference Model for evaluation
        self.z = z
        self.zm = zm
        self.zv = zv
        self.zm_prior = zm_prior
        self.zv_prior = zv_prior
        
        # Return list of k reconstructed batches
        px_batch = torch.cat([x_recon[i].unsqueeze(0) for i in range(self.mixtures_count)],dim=0)
        return px_batch
      
import numpy as np
import math

class NegVariationalLowerBound(nn.Module):
    
     def __init__(self, model, size_average = False, logger = Logger()):
        super(NegVariationalLowerBound, self).__init__()   
        self.vae = model
        self.size_average = size_average
        self.logger = logger
        
        # Detailed losses for output
        self.cond_entrop_qy = 0.
        self.recon_loss = 0.
        self.var_loss = 0.
        self.total_loss = 0.
        self.acc = 0.
        self.acc_counter = 0
        
        # Caches for accuracy computation
        self.qy = None
        
     def forward(self, netoutput, netinput):
        # Unpack output data
        x_recon = netoutput
        x = netinput.view(netinput.size(0),-1)
        
        # Get inference variables
        z, zm, zv = self.vae.z, self.vae.zm, self.vae.zv
        zm_prior, zv_prior = self.vae.zm_prior, self.vae.zv_prior
        
        # Get some parameters
        k = len(z)
        
        # propose distribution over y
        qy, qy_linear = self.vae.qy(x)
        
        # Neg-Entropy (qy) >> "ANTI CLUSTERING PRIOR"
        nent = torch.sum(qy * F.log_softmax(qy_linear, 1), 1)   
        
        losses = []
        sum_recon_loss = None
        sum_var_loss = None
        for i in range(k):
            # Reconstruction loss
            # ln p(x | y,z) >> INVERTED >> MINIMAZATION PROBLEM
            recon_loss = F.binary_cross_entropy(F.sigmoid(x_recon[i]), x, reduce=False) # [B]
            recon_loss = torch.sum(recon_loss, 1)

            # ln p(z | y) - ln(q(z | x,y)) >> INVERTED >> MINIMAZATION PROBLEM
            m_latent = torch.distributions.Normal(zm[i], zv[i])
            m_prior = torch.distributions.Normal(zm_prior[i], zv_prior[i])
            var_loss = torch.sum(m_latent.log_prob(z[i]) - m_prior.log_prob(z[i]), 1)    # [B]

            # ln p(y) (- q(y|x)) >> INVERTED >> MINIMIZATION PROBLEM
            # -q(y|x) is already inside of neg-entropy as qy * log(qy) == nent
            prior = -np.log(1 / k)
            
            # Add Log-losses
            losses.append(var_loss + recon_loss + prior)

            #save for print
            if sum_recon_loss is None:
                sum_recon_loss = recon_loss.sum()
                sum_var_loss = var_loss.sum()
            else:
                sum_recon_loss = recon_loss.sum() if sum_recon_loss > recon_loss.sum() else sum_recon_loss
                sum_var_loss = var_loss.sum() if sum_var_loss > var_loss.sum() else sum_var_loss

        # Expectation under q(y|x)
        loss = torch.sum(torch.cat([nent.unsqueeze(1) , (torch.stack(losses, 1) * qy)], 1))
        
        # Save for pretty-print (self.__repr__())
        self.cond_entrop_qy = -nent.sum().item() / netinput.size(0)
        self.recon_loss = sum_recon_loss.item() / netinput.size(0)
        self.var_loss = sum_var_loss.item() / netinput.size(0)
        self.total_loss = loss.sum().item() / netinput.size(0)
        self.qy = qy.data 
        
        # Return averaged or batch-aggregated loss
        if self.size_average == True:
            return loss.sum() / netinput.size(0)
        else:
            return loss.sum()
   
    
     def compute_special_losses(self, ctx, output, targets, loss):
         acc = NegVariationalLowerBound.compute_cluster_accuracy(self.qy.cpu(), targets)
         if not "cluster_accuracy" in ctx:
             self.acc = acc
             self.acc_counter = 1
         else:
             self.acc_counter = self.acc_counter + 1
             self.acc = ummon.utils.online_average(acc, self.acc_counter, self.acc)
         ctx["cluster_accuracy"] = self.acc
         ctx["H(y|x)"] = self.cond_entrop_qy
         ctx["reconstruction_loss"] = self.recon_loss
         ctx["variational_loss"] = self.var_loss
         ctx["total_loss"] = self.total_loss
         return ctx
    
     @staticmethod
     def compute_cluster_accuracy(qy, targets):
         ks = qy.size(1)
         targets = targets.cpu()
         cat_pred = Predictor.classify(qy)    # [B, 1]
         real_pred = np.zeros_like(cat_pred)  # [B, 1]
         for k in range(ks):
             # Find label of sample that belongs most to cluster k
             idx = qy[:,k].max(0, keepdim=True)[1]
             lab = targets[idx]               # scalar
             # Assing that label to all others from the cluster
             if len(lab) == 0:
                 continue
             real_pred[cat_pred.numpy() == k] = lab
         acc = np.mean(targets.unsqueeze(1).numpy() == real_pred)
         return acc
    
     def __repr__(self):
         return "NegVariationalLowerBound()"
            
