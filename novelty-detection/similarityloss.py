import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityLoss(nn.Module):
    """
    Similarity loss as an alternative to a pixel-wise loss.
    
    Typical usage is:
        y_synth = model.forward(x)
        y_real  = x
        
        sim_model = SimilarityModel()
        loss = SimilarityLoss(sim_model, norm='L2').forward(y_synth, y_real)
    
    """
    def __init__(self, model, train_model=False, p_norm = 2, layer = -1, size_average = False, reduce=True):
        super(SimilarityLoss, self).__init__()   
        self.model = model
        self.size_average = size_average
        self.reduce = reduce
        self.p = p_norm
        self.layer = layer

        if train_model == False:
            for param in self.model.parameters():
                param.requires_grad = False
        
    def forward(self, X, y):
        assert X.size(0) == y.size(0)
        B = X.size(0)

        if self.layer != -1:
            descr_a = self.early_stop(X, self.model, self.layer)
            descr_b = self.early_stop(y, self.model, self.layer)
        else:
            descr_a = self.model(X)
            descr_b = self.model(y)
        
        loss = F.pairwise_distance(descr_a, descr_b, p=self.p) # [B , 1]
        
        if self.reduce:
            if self.size_average == True:
                 return loss.sum() / B
            else:
                 return loss.sum()
        else:
            return loss.view(-1)           
        
        
    def early_stop(self, x, model, layer):
        layercount = 0    
        for name, layer in model._modules.items():
            x = layer(x)
            if layercount == layer:
                return x