#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:19:26 2018

@author: matthias
"""

 #class SimpleClassifier(nn.Module):
 #   def __init__(self, debug = False):
 #       super(SimpleClassifier, self).__init__()
 #       self.affine1 = nn.Linear(GLIMPSE_SIZE*GLIMPSE_SIZE*GLIMPSE_LEVELS, 256) 
 #       self.affine2 = nn.Linear(256, 256)
 #       self.affine3 = nn.Linear(256, 10)
 #       
 #       # Debug
 #       self.debug = debug
 #       
 #   def forward(self, data_batch):
 #       
 #       if self.debug:
 #            print("Data batch:", data_batch.shape, data_batch[0])
 #       
 #       B = data_batch.size(0)
 #                                                              # glimpse [Batch, Depth, Channels, Heigh, Width]
 #       px = data_batch.view(B, -1)                            # Flatten to column vector
 #       y1 = F.relu(self.affine1(px))
 #       y2 = F.relu(self.affine2(y1))       
 #       scores = F.softmax(self.affine3(y2), dim=1)
 #       return scores 
                     
