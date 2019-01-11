import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

class BasicLoss(nn.Module):
     
     def __init__(self, model, size_average = False, reduce=True, *args, **kwargs):
        super(BasicLoss, self).__init__()   
        self.model = model
        self.size_average = size_average
        self.reduce = reduce
        
     def _return_loss(self, loss):
         # Return averaged or batch-aggregated loss
         B = loss.size(0)
         if self.reduce:
             if self.size_average == True:
                 return loss.sum() / B
             else:
                 return loss.sum()
         else:
             return loss
             

class KLLoss(BasicLoss):
    
     def __init__(self, mean = 0, scale = 1, *args, **kwargs):
        super(KLLoss, self).__init__(*args, **kwargs)   
        
        self.mean = mean
        self.scale = scale
        
     def _variational_loss(self):
        # ln p(z | y) - ln(q(z | x,y)) >> INVERTED >> MINIMAZATION PROBLEM
        m_latent = torch.distributions.Normal(self.model.m, self.model.v)
        m_prior = torch.distributions.Normal(torch.zeros_like(self.model.z) + self.mean, self.scale * torch.ones_like(self.model.z))
        return m_latent.log_prob(self.model.z) - m_prior.log_prob(self.model.z)
        
     def _reconstruction_loss_bernoulli(self, inpt, output):
        # Reconstruction loss
        return F.binary_cross_entropy(F.sigmoid(output), inpt, reduce=False)
     
     def _reconstruction_loss_normal(self, inpt, output):
        # Reconstruction loss
        return (output - inpt)**2
        
     def forward(self, netoutput, inpt):
        M = netoutput.size(1)
        B = netoutput.size(0)
        
        loss = self._variational_loss().sum(dim=1)                              # B
        loss += self._reconstruction_loss_normal(inpt, self.model.o).sum(dim=1) # B
        
        return self._return_loss(loss)
     
     def __repr__(self):
         return "KLLoss()"
   
class KLLoss_lognormal(KLLoss):
    
     def __init__(self, *args, **kwargs):
        super(KLLoss_lognormal, self).__init__(*args, **kwargs)   
     
     def _lognormal_variational_loss(self):
        # ln p(z | y) - ln(q(z | x,y)) >> INVERTED >> MINIMAZATION PROBLEM
        m_latent = torch.distributions.LogNormal(self.model.m, self.model.v)
        m_prior = torch.distributions.LogNormal(torch.zeros_like(self.model.z) + self.mean, self.scale * torch.ones_like(self.model.z))
        return m_latent.log_prob(self.model.z) - m_prior.log_prob(self.model.z)
        
     def forward(self, netoutput, inpt):
        loss = self._lognormal_variational_loss().sum(dim=1)                         # B x 1
        loss += self._reconstruction_loss_normal(inpt, self.model.o).sum(dim=1)      # B x 1
        
        return self._return_loss(loss)                            # 1 x 1
        
     def __repr__(self):
         return "KLLoss_lognormal()"
     
        
class KLLoss_laplace(KLLoss):
    
     def __init__(self, *args, **kwargs):
        super(KLLoss_laplace, self).__init__(*args, **kwargs)   
     
     def _laplace_variational_loss(self):
        # ln p(z | y) - ln(q(z | x,y)) >> INVERTED >> MINIMAZATION PROBLEM
         m_latent = torch.distributions.Laplace(self.model.m, self.model.v)
         m_prior = torch.distributions.Laplace(torch.zeros_like(self.model.z) + self.mean, self.scale * torch.ones_like(self.model.z))
         return m_latent.log_prob(self.model.z) - m_prior.log_prob(self.model.z)
        
     def forward(self, netoutput, inpt):
        loss = self._laplace_variational_loss().sum(dim=1)                           # B x 1
        loss += self._reconstruction_loss_normal(inpt, self.model.o).sum(dim=1)      # B x 1
        
        return self._return_loss(loss)
        
     def __repr__(self):
         return "KLLoss_laplace()"        


class NonGaussianRegularizer(BasicLoss):
    
    def __init__(self, *args, **kwargs):
         super(NonGaussianRegularizer, self).__init__(*args, **kwargs)   
    
    def forward(self, netoutput, inpt):
        with_cuda = next(self.model.parameters()).is_cuda
        B = netoutput.size(0)
        M = netoutput.size(1)
    
        noise = np.random.normal(0,1,M*B).reshape(-1, M).astype(np.float32)
        noise = torch.from_numpy(noise)
        if with_cuda:
            noise = noise.cuda()
        noise_t = Predictor.predict(self.model, noise)
        loss_non_gauss = -0.01*(self.model.z - noise_t)**2
    
        return self._return_loss(loss_non_gauss.sum(dim=1))
     
    def __repr__(self):
     return "NonGaussianRegularizer()"
 
class NonUniformRegularizer(BasicLoss):
    
    def __init__(self, *args, **kwargs):
         super(NonUniformRegularizer, self).__init__(*args, **kwargs)   
    
    def forward(self, netoutput, inpt):
        with_cuda = next(self.model.parameters()).is_cuda
        B = netoutput.size(0)
        M = netoutput.size(1)
    
        noise = np.random.uniform(0,1,M*B).reshape(-1, M).astype(np.float32)
        noise = torch.from_numpy(noise)
        if with_cuda:
            noise = noise.cuda()
        noise_t = Predictor.predict(self.model, noise)
        loss_non_gauss = -0.01*(self.model.z - noise_t)**2
    
        return self._return_loss(loss_non_gauss.sum(dim=1))
     
    def __repr__(self):
     return "NonUniformRegularizer()"

from torch.autograd import Variable
class KLLoss_shift_r(BasicLoss):
    
    def __init__(self, left = True, *args, **kwargs):
         super(KLLoss_shift_r, self).__init__(*args, **kwargs)   
         self.left = left
        
    def forward(self):
        percentile = 10
        true_r = Variable(torch.from_numpy(np.array(np.percentile(self.model.z.data.cpu().numpy(), percentile))).float())
        r = true_r
        
        if self.left:
            self.model.z = self.model.z - r
        else:
            self.model.z = self.model.z + r
            
        return 0.
        
    def __repr__(self):
        return "KLLoss_shift_r()"     
    
    
class KLLoss_cut_negative(BasicLoss):
    
    def __init__(self, *args, **kwargs):
         super(KLLoss_cut_negative, self).__init__(*args, **kwargs)    
        
    def forward(self, loss):
        loss = loss * (self.model.z > 0).float()
        return self._return_loss(loss)
        
    def __repr__(self):
        return "KLLoss_cut_negative()" 
    

from similarityloss import SimilarityLoss
class KLLoss_laplace_with_similarity_loss(KLLoss):

     def __init__(self, model, train_similarity_model=False, *args, **kwargs):
         super(KLLoss_laplace_with_similarity_loss, self).__init__(model=model, *args, **kwargs)   
         assert hasattr(model, "similarity")
         model_similarity = model.similarity
         self.similarityloss = SimilarityLoss(model_similarity, train_model=train_similarity_model, size_average=self.size_average, reduce=False)
     
     def _laplace_variational_loss(self):
        # ln p(z | y) - ln(q(z | x,y)) >> INVERTED >> MINIMAZATION PROBLEM
         m_latent = torch.distributions.Laplace(self.model.m, self.model.v)
         m_prior = torch.distributions.Laplace(torch.zeros_like(self.model.z) + self.mean, self.scale * torch.ones_like(self.model.z))
         return m_latent.log_prob(self.model.z) - m_prior.log_prob(self.model.z)
        
     def _similarity_loss(self, inpt, outpt):
         return self.similarityloss(inpt, outpt)
    
     def forward(self, netoutput, inpt):
        loss = self._laplace_variational_loss().sum(dim=1)                           # B x 1
        loss += self._reconstruction_loss_normal(inpt, self.model.o).sum(dim=1)      # B x 1
        loss += self._similarity_loss(inpt, self.model.o)                            # B x 1
        return self._return_loss(loss)
        
     def __repr__(self):
         return "KLLoss_laplace_with_similarity_loss()"   
     
        
        
from ganloss import GanLoss
class KLLoss_laplace_with_gan_loss(KLLoss):

     def __init__(self, model, *args, **kwargs):
         super(KLLoss_laplace_with_gan_loss, self).__init__(model=model, *args, **kwargs)   
         assert hasattr(model, "descriminator")
         self.ganloss = GanLoss(model.descriminator, size_average=self.size_average, reduce=False)
         self.loss_var = 0
         self.loss_rec = 0
         self.loss_gan = 0
     
     def _laplace_variational_loss(self):
        # ln p(z | y) - ln(q(z | x,y)) >> INVERTED >> MINIMAZATION PROBLEM
         m_latent = torch.distributions.Laplace(self.model.m, self.model.v)
         m_prior = torch.distributions.Laplace(torch.zeros_like(self.model.z) + self.mean, self.scale * torch.ones_like(self.model.z))
         return m_latent.log_prob(self.model.z) - m_prior.log_prob(self.model.z)
        
    
     def forward(self, netoutput, inpt):
        loss = self._laplace_variational_loss().sum(dim=1)                           # B x 1
        self.loss_var = loss.sum()
        loss += self._reconstruction_loss_normal(inpt, self.model.o).sum(dim=1)      # B x 1
        self.loss_rec = loss.sum() - self.loss_var
        loss += self.ganloss(inpt, self.model.o)                                     # B x 1
        self.loss_gan = loss.sum() - self.loss_var - self.loss_rec
        return self._return_loss(loss)
        
     def __repr__(self):
         return "KLLoss_laplace_with_gan_loss(): variational: {:.2f} recon: {:.2f} gan: {:.2f}".format(self.loss_var, self.loss_rec, self.loss_gan)   
 