#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REINFORCE Implementation

[1] Williams, R. J. (1992). 
    Simple statistical gradient-following methods for connectionist reinforcement learning. 
    Machine Learning, 8, 229–256. https://doi.org/10.1007/BF00992696


Created on Mon Jan 15 13:01:27 2018

@author: matthias
@copyright: IOS
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable

class GlimpseNetwork(nn.Module):
    def __init__(self, hidden_neurons_hg, hidden_neurons_hl, hidden_neurons_hs, output_neurons, args, debug=False, verbose=False):
        super(GlimpseNetwork, self).__init__()    
        
        self.args = args   
        self.debug = debug
        self.verbose = verbose

        self.affine1 = nn.Linear(args.glimpse_size*args.glimpse_size*args.glimpse_level, hidden_neurons_hg, bias=False)
        self.affine2 = nn.Linear(2, hidden_neurons_hl, bias=False)
        self.affine3 = nn.Linear(hidden_neurons_hg, output_neurons, bias=False)
        self.affine4 = nn.Linear(hidden_neurons_hl, output_neurons, bias=False)

        if self.args.learn_to_scale:               
            self.affine5 = nn.Linear(1, hidden_neurons_hs, bias=False)
            self.affine6 = nn.Linear(hidden_neurons_hs, output_neurons, bias=False)

        
    def forward(self, glimpse, glimpse_location, glimpse_scale):      
        B = glimpse.size(0)
        px = glimpse.view(B, -1)                               # Flatten to column vector
        hg = F.relu(self.affine1(px))
        hl = F.relu(self.affine2(glimpse_location))            # glimpse_location [B, x, y]
        
        if self.args.learn_to_scale:
            hs = F.relu(self.affine5(glimpse_scale))
            g = F.relu(self.affine3(hg) + self.affine4(hl) + self.affine6(hs))
        else:
             g = F.relu(self.affine3(hg) + self.affine4(hl))

        if self.debug:
            self.gg = glimpse
            self.l = glimpse_location
            self.hg = hg
            self.hl = hl
            self.g = g
            if self.verbose:
                print("---GlimpseNetwork---")
                print("hg-sum", torch.sum(hg, dim=1))
                print("glimpse", glimpse)
                print("glimpse-sum", torch.sum(torch.sum(torch.sum(glimpse, dim=1), dim=1),dim=1))
                print("glimpse_location", glimpse_location)
                print("hl-sum", torch.sum(hl, dim=1))
                print("hl-weights", self.affine2.weight)
                print("g-sum", torch.sum(g, dim=1))        
        return g 
    
class InternalState(nn.Module):
    def __init__(self, glimpse_representation_neurons, hidden_state_neurons, args, debug=False):
        super(InternalState, self).__init__()

        self.debug = debug
        self.use_lstm = args.use_lstm
        self.use_cuda = args.use_cuda
        self.hidden_state_neurons = hidden_state_neurons        
        
        if not self.use_lstm:
            self.affine1 = nn.Linear(hidden_state_neurons, hidden_state_neurons, bias=False)
            self.affine2 = nn.Linear(glimpse_representation_neurons, hidden_state_neurons, bias=False)     
        else:
            self.lstm = nn.LSTMCell(hidden_state_neurons, hidden_state_neurons)

    def init_hidden_state(self, batch_size):
        if self.use_cuda:
            return Variable(torch.DoubleTensor(batch_size, self.hidden_state_neurons).zero_().float()).cuda()
        return Variable(torch.DoubleTensor(batch_size, self.hidden_state_neurons).zero_().float())
    
    def forward(self, glimpse_representation, hidden_state=(None, None)):
        hidden_s =  hidden_state[0]
        cell_s = hidden_state[1]    
        
        if self.use_lstm == False:
            if hidden_s is None:
                hidden_s = self.init_hidden_state(glimpse_representation.size(0))
            hg = self.affine2(glimpse_representation)        
            hh = self.affine1(hidden_s)
            ht = F.relu(hh + hg)   
            hc = None
            if self.debug:
                self.hh = hh
                self.hg = hg
                self.ht = ht  
        else:
            if hidden_s is None:
                hidden_s = self.init_hidden_state(glimpse_representation.size(0))
                cell_s = self.init_hidden_state(glimpse_representation.size(0))
            ht, hc = self.lstm(glimpse_representation, (hidden_s, cell_s))

            if self.debug:
                self.ht = ht  
                self.hg = hc
        return (ht, hc)
 
    
class ActionNetwork(nn.Module):
    def __init__(self, hidden_state_neurons, classes, debug=False, verbose=False):
        super(ActionNetwork, self).__init__()

        self.debug = debug
        self.verbose = verbose
        
        self.affine1 = nn.Linear(hidden_state_neurons, classes, bias=False)  
  
    def forward(self, hidden_state):
        y = self.affine1(hidden_state)
        if self.verbose:
            print("----ActionNetwork----")
            print("hidden-state", hidden_state)
            print("y", y)
        return y
    
    
class LocationNetwork(nn.Module):
    def __init__(self, input_neurons, output_neurons, debug=False):
        super(LocationNetwork, self).__init__()

        self.affine1 = nn.Linear(input_neurons, output_neurons, bias=False)        
        
    def forward(self, hidden_state):
        y1 = F.tanh(self.affine1(hidden_state))
        return y1 
    
class ScaleNetwork(nn.Module):
    def __init__(self, input_neurons, output_neurons, debug=False):
        super(ScaleNetwork, self).__init__()

        self.affine1 = nn.Linear(input_neurons, output_neurons, bias=False)        
        
    def forward(self, hidden_state):
        hh = self.affine1(hidden_state)
        y = F.softmax(hh, dim=1)
        return y # [SCALE_0, SCALE_1, SCALE_2] 
    
class ValueNetwork(nn.Module):
    def __init__(self, input_neurons, output_neurons, debug=False):
        super(ValueNetwork, self).__init__()
        
        self.affine1 = nn.Linear(input_neurons, output_neurons, bias=False)        
        
    def forward(self, hidden_state):
        y1 = F.sigmoid(self.affine1(hidden_state))
        return y1 

class Policy(nn.Module):
    def __init__(self, hidden_neurons, args = None, debug = False):
        super(Policy, self).__init__()
       
        self.args = args

        self.location_network = LocationNetwork(hidden_neurons, 2)
        self.value_network = ValueNetwork(hidden_neurons, 1)
        if args.learn_to_scale:       
            self.scale_network = ScaleNetwork(hidden_neurons, 3)   
    
        self.saved_ln_pis = []
        self.saved_baselines = []
        self.rewards = []
    
        self.debug = debug
        
    def forward(self, hidden_state):
        baseline = self.value_network(hidden_state)
        location_scores = self.location_network(hidden_state)
        if self.args.learn_to_scale:
            scale_scores =  self.scale_network(hidden_state)
        else:
            scale_scores = None
        return location_scores, scale_scores, baseline
 

def get_glimpse(glimpse_sensor, glimpse_location, glimpse_scale, init_glimpse_rand, init_scale_rand):
    
     # INIT GLIMPSES
     if glimpse_location is None:
        if init_glimpse_rand:
            glimpse_location = np.random.uniform(-1,1,(glimpse_sensor.IM.shape[0], 2))
        else:
            glimpse_location = np.zeros((glimpse_sensor.IM.shape[0], 2))                
        glimpse_location = Variable(torch.from_numpy(glimpse_location).float())
        
     if glimpse_scale is None:
        if init_scale_rand:
            glimpse_scale = np.random.uniform(0,2,(glimpse_sensor.IM.shape[0], 1))
        else:
            glimpse_scale = np.zeros((glimpse_sensor.IM.shape[0], 1))    
        glimpse_scale = Variable(torch.from_numpy(glimpse_scale).int().float())
     
     pyramide, reward, _, _ =  glimpse_sensor.step(glimpse_location.data.numpy(), glimpse_scale.data.int().numpy())
     
     if pyramide.dtype != "float32":
         print("Glimpse was not float32, was", pyramide.dtype)
         pyramide = pyramide.astype("float32")
     
     if np.max(pyramide) > 1.:
         print("Glimpse was not scaled between 0 and 1")
         pyramide = pyramide / 255.
    
     glimpse = torch.from_numpy(pyramide).float()
     return glimpse, reward, glimpse_location, glimpse_scale    
    
import ummon.utils as uu
    
class RVA(nn.Module):
    def __init__(self, args, glimpse_sensor = None, debug = False, verbose=False):
        super(RVA, self).__init__()

        self.args = args
        self.use_cuda = args.use_cuda
        self.debug = debug
        self.verbose = verbose
        self.is_train = True
        self.handles = []

        self.glimpse_sensor = glimpse_sensor
        self.glimpse_network = GlimpseNetwork(hidden_neurons_hg=128, hidden_neurons_hl=128, hidden_neurons_hs=128, output_neurons=256, args=args, debug=debug)
        self.internal_state = InternalState(glimpse_representation_neurons=256, hidden_state_neurons=256, args=args, debug=debug)
        self.action_network = ActionNetwork(hidden_state_neurons=256, classes=10, debug=debug)
        self.policy = Policy(hidden_neurons=256, args=args, debug=debug)
        
        #History
        self.classifier_scores_history = []
        self.glimpses_location_history = [] 
        self.glimpses_scale_history = []
           
            
        def weights_init_xavier(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
               
        def weights_init_eye(m):
             if type(m) == nn.Linear:
                nn.init.eye_(m.weight)
                m.weight.requires_grad = True

        def weights_init_normal(m):
            if type(m) == nn.Linear:
              nn.init.normal_(m.weight, mean=0, std=0.1)
        
        # Init weights
        self.internal_state.apply(weights_init_eye)  
    
    def train(self, mode=True):
        super(RVA, self).train()
        self.is_train = True
        uu.unregister_hooks_(self.handles)
        
    def eval(self):
        super(RVA, self).eval()
        self.is_train = False
        self.handles = uu.register_nan_checks_(self)
    
    def forward(self, inputs, train=None):
        if next(self.parameters()).is_cuda != self.use_cuda:
            self.cuda() if self.use_cuda else self.cpu()
        
        # Cleanup
        if train is not None:
            self.is_train = train
        del self.policy.rewards[:]
        del self.policy.saved_ln_pis[:]              
        del self.policy.saved_baselines[:]
        del self.classifier_scores_history[:]
        del self.glimpses_location_history[:]
        del self.glimpses_scale_history[:]          
            
        action = True if (self.args.glimpse_count == 1) or not self.is_train else False
        action_scores, hidden_state, (glimpse_location, glimpse_scale), reward = self.step(inputs, hidden_state = (None, None), action=action)        

        if not self.is_train:
            # bookkeeping for log
            self.glimpses_location_history.append(glimpse_location)
            self.glimpses_scale_history.append(glimpse_scale)
            self.classifier_scores_history.append(action_scores)     
        
        for t in range(1, self.args.glimpse_count):  # Allow GLIMPSES_COUNT Glimpses per episode
            
            # Execute Model
            action = True if (t == self.args.glimpse_count - 1) or not self.is_train else False
            action_scores, hidden_state, (glimpse_location, glimpse_scale), reward = self.step(None, hidden_state, action=action)
        
            if not self.is_train:
                # bookkeeping for log
                self.glimpses_location_history.append(glimpse_location)
                self.glimpses_scale_history.append(glimpse_scale)
                self.classifier_scores_history.append(action_scores)     
                    
            # Append reward
            self.policy.rewards.append(reward)
            
        return action_scores
          
    def step(self, inputs, hidden_state=(None,None), action=True):
        
        # INITIALIZE GLIMPSE SENSOR
        if not inputs is None:
            self.glimpse_sensor.load_batch(inputs.cpu().data.numpy())
        
        # EXECUTE POLICY
        if hidden_state[0] is not None:
            if self.args.use_location_rnn:
                location_scores, scale_scores, baseline = self.policy(hidden_state[0])   
            else:
                location_scores, scale_scores, baseline = self.policy(hidden_state[0].detach())
            
            # NORMAL DISTRIBUTION
            if self.is_train==True:
                eps = (np.random.normal(0, np.sqrt(self.args.location_variance), size=location_scores.data.shape)).astype(np.float32)
                eps = torch.from_numpy(eps)
            else:
                eps = 0.
            glimpse_location = Variable(location_scores.cpu().data + eps)
          
            # COMPUTE LOG PROBABILITY
            ln_pi = -0.5 * torch.sum((glimpse_location - location_scores.cpu())*(glimpse_location - location_scores.cpu()),dim=1) / self.args.location_variance
          
            if self.args.learn_to_scale:
                # CATEGORIAL DISTRIBUTION
                m = Categorical(scale_scores.cpu())
                if self.is_train==True:
                    scale_actions = m.sample()     
                else:
                    scale_actions = scale_scores.cpu().max(1)[1]
            
                glimpse_scale = Variable(scale_actions.data.unsqueeze(1).float())
                
                ln_pi = ln_pi + m.log_prob(scale_actions)
            else:
                glimpse_scale = None
                
            # SAVE LOG PROPABILITY FOR REWARD
            self.policy.saved_ln_pis.append(ln_pi.unsqueeze(1))
            self.policy.saved_baselines.append(baseline.cpu().unsqueeze(1))
      
            # FETCH GLIMPSE FROM SENSOR
            glimpse, reward, glimpse_location, glimpse_scale = get_glimpse(self.glimpse_sensor, glimpse_location, glimpse_scale, init_glimpse_rand = self.args.init_glimpse_rand and self.is_train, init_scale_rand = self.args.learn_to_scale and self.is_train)
            
        else:
            # FETCH GLIMPSE FROM SENSOR
            glimpse, reward, glimpse_location, glimpse_scale = get_glimpse(self.glimpse_sensor, glimpse_location = None, glimpse_scale = None, init_glimpse_rand = self.args.init_glimpse_rand and self.is_train,  init_scale_rand = self.args.learn_to_scale and self.is_train)
      
        
        # COPY GLIMPSE TO CUDA
        if self.use_cuda:
            glimpse_location = glimpse_location.cuda()
            glimpse_scale = glimpse_scale.cuda()
            glimpse = glimpse.cuda()
      
        # COMPUTE GLIMPSE
        y1 = self.glimpse_network(Variable(glimpse), glimpse_location, glimpse_scale)
        
        # COMPUTE HIDDEN STATE
        hidden_state, cell_state = self.internal_state(y1, hidden_state)
        
        # COMPUTE OUTPUT
        if action:
            action_scores = self.action_network(hidden_state)
            action_scores = action_scores
        else:
            action_scores = None
        
        return action_scores, (hidden_state, cell_state), (glimpse_location, glimpse_scale),  torch.from_numpy(reward).float()


 

