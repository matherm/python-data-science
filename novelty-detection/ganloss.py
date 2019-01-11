import torch
import torch.nn as nn
import torch.nn.functional as F

class GanLoss(nn.Module):
    """
    Gan loss as an alternative to a pixel-wise loss.
    
    Typical usage is:
        y_synth = model.forward(x)
        y_real  = x
        
        sim_model = GanModel()
        loss = GanLoss(sim_model).forward(y_real, y_synth)
    
    """
    def __init__(self, disc_model, size_average = False, reduce=True):
        super(GanLoss, self).__init__()   
        self.size_average = size_average
        self.reduce = reduce
        self.discriminator = disc_model
        self.nll = nn.BCEWithLogitsLoss(size_average=self.size_average, reduce=self.reduce)
        
    def forward(self, inpt, outp):
        reals, fakes = inpt, outp
        B = fakes.size(0)
        ####
        # Labels
        ####
        valid_labels = torch.Tensor(torch.ones(reals.size(0)).unsqueeze(1))
        fake_labels  = torch.Tensor(torch.zeros(fakes.size(0)).unsqueeze(1))
        
        ####
        # Generator
        ####
        for param in self.discriminator.parameters():
            param.requires_grad = False
        fakes_preds = self.discriminator(fakes)      
        g_loss = self.nll(fakes_preds, valid_labels)
     
        ####
        # Discriminator
        ####
        if self.reduce == False: assert reals.size(0) == fakes.size(0)
        for param in self.discriminator.parameters():
            param.requires_grad = True
        preds_reals = self.discriminator(reals)
        preds_fakes = self.discriminator(fakes)
        d_reals_loss = self.nll(preds_reals, valid_labels)
        d_fakes_loss = self.nll(preds_fakes, fake_labels)
        
        loss =  d_reals_loss +  d_fakes_loss + g_loss        
        
        if self.reduce:
            if self.size_average == True:
                return loss.sum() / B
            else:
                return loss.sum()
        else:
            return loss.view(-1)
