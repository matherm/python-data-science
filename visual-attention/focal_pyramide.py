#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:39:30 2018

@author: matthias
"""

import numpy as np
import scipy

def comp_focal_pyramide(IM, focus, gh, gw, levels, scale, init_scale, debug=False):
    """
    Generates a focal pyramide 
    
    Paramaters
    ----------
    IM          are Images ([Batch, Channels, Height, Width])
    focus       are coordinates ([Batch, 2 (x (float) ,y (float))]) x, y e [-1 , +1]           
    gh          is  window height
    gw          is  window width
    levels      is  pyramide depths
    scale       is  scale between each level
    init_scale  is  ([Batch, 1 (int)]) initial scale of the glimpse (for levels=2 and scale=0, 2^1, and 2^2 is returned)
    
    
    Return:
    -------
    Focal Pyramide ([Batch, levels, Channels, gh, gw])
    """
    B = IM.shape[0]
    C = IM.shape[1]
    H = IM.shape[2]
    W = IM.shape[3]
    image_max_x = H - 1
    image_may_y = W - 1
         
    P = np.zeros((B, levels, C, gh, gw), dtype=IM.dtype) 
    # FOR EVERY IMAGE IN BATCH
    for bb in range(B):
        # COMPUTE PIXEL LOCATION FROM RANGE [-1, +1]
        x = int((focus[bb, 0] + 1)  * image_max_x/2)
        y = int((focus[bb, 1] + 1)  * image_may_y/2)
        # LEVEL BY LEVEL
        for ll in range(init_scale[bb][0], levels+init_scale[bb][0], 1):
            scale_local = scale**ll
            gh_local = gh * scale_local
            gw_local = gw * scale_local
            # COMPUTE POSITIONS
            u = x-int(gh_local/2)
            b = x+int(gh_local/2)
            l = y-int(gw_local/2)
            r = y+int(gw_local/2)
            # DEFINING OFFSETS
            u_off = 0
            l_off = 0
            # HANDLING BOUNDARY
            if debug:
                print("u", u) 
                print("b", b)
                print("l", l) 
                print("r", r)  
            if u < 0:
                u_off = int(np.floor(-u * 1./scale_local))
                u = 0
                if b < 0:
                    break # ROI totally outside                
            if b >= H:
                b = H
                if u >= H:
                    break # ROI totally outside
            if l < 0:
                l_off = int(np.floor((-l) * 1./scale_local))
                l = 0
                if r < 0:
                    break # ROI totally outside
            if r >= W:
                r = W
                if l >= W:
                    break # ROI totally outside            
            IM_resized = scipy.ndimage.zoom(IM[bb, :, u:b, l:r], (1, 1./scale_local, 1./scale_local) , order=0)      
            im_height = IM_resized.shape[1]
            im_width  = IM_resized.shape[2]  
            if debug:
                print("u", u) 
                print("b", b)
                print("l", l) 
                print("r", r) 
                print("inverse_scale_local", 1./scale_local)
                print("u_off", u_off)
                print("l_off", l_off)    
            P[bb, ll, :, u_off:u_off+im_height, l_off:l_off+im_width] = IM_resized
            
    return P  

def comp_focal_pyramide_opt(IM, focus, size, levels=1, init_scale=None):
    """
    Generates a focal pyramide (optimized version only support quadratic patches)
    
    Paramaters
    ----------
    IM           are Images ([Batch, Channels, Height, Width])
    focus        are coordinates ([Batch, 2 (x (float) ,y (float))]) x, y e [-1 , +1]        
    size         is  patch size
    levels       is  pyramide depths
    init_scale  is  ([Batch, 1 (int)]) initial scale of the glimpse (for levels=2 and scale=0, 2^1, and 2^2 is returned)
    
    Return:
    -------
    Focal Pyramide ([Batch, levels, Channels, gh, gw])
    """
    
    B = IM.shape[0]
    C = IM.shape[1]
    size_i = np.asarray(IM[0].shape[2:4])   

    # SORT BATCH BY INIT_SCALE
    if init_scale is None:
        scales = [0]
        init_scale = np.zeros((B,1), dtype="int32")
    else:
        scales = np.unique(init_scale)
        
    P = np.zeros((B, levels, C, size, size), dtype=IM.dtype) 
    
    assert init_scale.dtype == "int32"
    
    
    # EVERY SCALE EXTRA
    for ss in range(scales.shape[0]):     # e.g. 3 .. [0, 1, 2]
        
        current_scale = scales[ss]     # e.g. 0
        mask = (init_scale == current_scale)[:,0]
        IM_ = IM[mask, :, :, :]
        focus_ = focus[mask, :]
        
        BB = IM_.shape[0]
        # LEVEL BY LEVEL
        for ll in range(current_scale, levels + current_scale, 1):
            
            # CROP PATCH
            size_o = np.array([size * (2 ** ll), size * (2 ** ll)])
            h_o, w_o = size_o
            
            y = np.zeros((BB, C, h_o, w_o), dtype=IM.dtype) 
        
            # [-1, 1]^2 -> [0, h_i]x[0, w_i]
            center = 0.5 * (focus_+1) * (size_i+1)
    
            # topleft: np.array[batch, [top, left]]
            topleft = center - 0.5*size_o
            topleft = np.round(topleft).astype(np.int32)
    
            tl_o = np.maximum(topleft, 0)
            br_o = np.minimum(topleft+size_o, size_i)
    
            tl_i = tl_o - topleft
            br_i = br_o - topleft
    
            for k in range(BB):
                if (br_i[k,0] - tl_i[k,0]) < 0 or (br_i[k,1] - tl_i[k,1]) < 0:
                    continue
                y[k,:,tl_i[k,0]:br_i[k,0],tl_i[k,1]:br_i[k,1]] \
                    += IM_[k,:,tl_o[k,0]:br_o[k,0],tl_o[k,1]:br_o[k,1]]
            
            # POOL PATCH
            pooled = scipy.ndimage.zoom(y, (1, 1, 1./(2 ** ll), 1./(2 ** ll)) , order=0)      
            
            # CONCATENATE OUTPUT
            P[mask, ll - current_scale, :, :, : ] = pooled
            #P = np.concatenate((P, pooled), axis=1)     
        
    return P 