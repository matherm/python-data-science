#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:51:49 2018

@author: matthias
"""
import time
import numpy as np

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print("[{}]".format(self.name))
        print("Elapsed: {:.4f} ms".format(time.time() - self.tstart))
        
def plot_parameters(model, env):
    print_abs_avg_val("action-network grad", model.action_network.affine1.weight.grad) 
    print_abs_avg_val("action-network weight", model.action_network.affine1.weight)
    print_abs_avg_val("value-network grad", model.policy.value_network.affine1.weight.grad)
    print_abs_avg_val("value-network weight", model.policy.value_network.affine1.weight)
    print_abs_avg_val("loc-network grad", model.policy.location_network.affine1.weight.grad)
    print_abs_avg_val("loc-network weight", model.policy.location_network.affine1.weight)                    
    
    try: 
        print_abs_avg_val("state-network out g", model.internal_state.hg, env.batch_size)
        print_abs_avg_val("state-network out t", model.internal_state.ht, env.batch_size)
        print_abs_avg_val("state-network weight3", model.internal_state.lstm.weight_ih)
        print_abs_avg_val("state-network weight3", model.internal_state.lstm.weight_hh)
        print_abs_avg_val("state-network grad3", model.internal_state.lstm.weight_ih.grad)
        print_abs_avg_val("state-network grad3", model.internal_state.lstm.weight_hh.grad)
        print_abs_avg_val("state-network weight1", model.internal_state.affine1.weight)
        print_abs_avg_val("state-network weight2", model.internal_state.affine2.weight)
        print_abs_avg_val("state-network grad1", model.internal_state.affine1.weight.grad)
        print_abs_avg_val("state-network grad2", model.internal_state.affine2.weight.grad)
        print_abs_avg_val("state-network out h", model.internal_state.hh, env.batch_size)
    except AttributeError: 
        pass
    
    try: 
        print_abs_avg_val("glimpse-net grad1", model.glimpse_network.affine1.weight.grad)
        print_abs_avg_val("glimpse-net grad2", model.glimpse_network.affine2.weight.grad)
        print_abs_avg_val("glimpse-net grad3", model.glimpse_network.affine3.weight.grad)
        print_abs_avg_val("glimpse-net grad4", model.glimpse_network.affine4.weight.grad)
        print_abs_avg_val("glimpse-net weight1", model.glimpse_network.affine1.weight)
        print_abs_avg_val("glimpse-net weight2", model.glimpse_network.affine2.weight)
        print_abs_avg_val("glimpse-net weight3", model.glimpse_network.affine3.weight)
        print_abs_avg_val("glimpse-net weight4", model.glimpse_network.affine4.weight)
        print_abs_avg_val("glimpse-net output hl", model.glimpse_network.hl, env.batch_size)
        print_abs_avg_val("glimpse-net output hg", model.glimpse_network.hg, env.batch_size)
        print_abs_avg_val("glimpse-net output g", model.glimpse_network.g, env.batch_size)
        print_abs_avg_val("glimpse-net input l", model.glimpse_network.l, env.batch_size)
        print_abs_avg_val("glimpse-net input g", model.glimpse_network.gg, env.batch_size)
    except AttributeError:
        pass           

def print_abs_avg_val(name, var, batch_size=1):
    if not var is None:
            param_size = var.view(-1).size(0)
            val = var.abs().sum().data[0] / ( param_size * batch_size)
            print("{}: \t\t {:+5.12f} >0 => {}".format(name, val , val > 0.))    

def get_abs_avg_weights(model):
    weights = []
    for p in model.parameters():
        print(type(p))
        weights.append(p.data)
    return weights

def register_nan_checks(model):
    def check_grad(module, input, output):
        if not hasattr(module, "weight"):
            return
        if any(np.all(np.isnan(gi.cpu().data.numpy())) for gi in module.weight if gi is not None):
            print('NaN weights in ' + type(module).__name__)
        if any(np.all(gi.cpu().data.numpy() > 1.) for gi in module.weight if gi is not None):
            print('Exploding weights in ' + type(module).__name__)
    handles = []
    for module in model.modules():
        handles.append(module.register_forward_hook(check_grad))
    return handles
            
def unregister_hook(handles):
    for handle in handles:
        handle.remove()

if __name__ == "__main__":
    
    
    with Timer("abc"):
        i = 1000**2