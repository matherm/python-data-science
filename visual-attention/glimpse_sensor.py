#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import focal_pyramide as focal

"""
Implments a GlimpseSensor and an Environment allowing an Agent (Reinforcement-Learning) to interact with its environment.
Typical methods are ~step()~ and ~next_episode()~ where the former performs an step within an episode and the latter one skips the episode iteself.

Typical usage is:
    glimpse_sensor = GlimpseSensor(images, targets, batch_size=args.batch_size, loop=True, random=True, glimpse_size=(args.glimpse_size, args.glimpse_size), levels=(args.glimpse_level, 2), debug=False, optimized=True)                
  
    # Re-positioning the sensor
    pyramide, _, __, ___  = glimpse_sensor.step(new_location)
    
    # Skip to next episode (aka load new mini-batch of images)
    glimpse_sensor.next_episode()
"""

class GlimpseSensor():
    """
    Implements a vision environment that computes arbitrary glimpses of a static environment.
    
    See:
    [1] Mnih, V., Heess, N., Graves, A., & Kavukcuoglu, K. (2014). 
        Recurrent Models of Visual Attention. In Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, & K. Q. Weinberger (Eds.), 
        Advances in Neural Information Processing Systems 27: Annual Conference on Neural Information Processing Systems 2014, December 8-13 2014, Montreal, Quebec, Canada (pp. 2204–2212). 
        https://doi.org/ng
    
    The environment is initialized with a static image represented as a 3D-Tensor [H, W, C] where H = height, W = width and C are channels.
    The initial state is the center of the image.
    Edges are padded with zeroes.
    
    Parameters
    ----------
    glimpse_size as (GH, GW)
    levels       as (D, S) specifying the depths D of the returned pyramid and the scale factor S between each level
    dtype        as (Numpy.dtype)
    
    States
    ------
    IM current image
    x, y coordinates in the input space (image) e [-1, +1]
    glimpse, representing the current focal pyramide [D, C, GH, GW]
    
    Actions
    -----
    y, x [Batch, 2 (float, float)] coordinates [-1, +1] where the new glimpse shall be centered
    
    @author: matthias hermann
    @copyright IOS
    """
    
    def __init__(self, glimpse_size=(0,0), levels=0, enable_history=False, debug=False, optimized=True):
             
        #STORING ENVIRONMENT
        self.i = 0
        self.swap_channel_major = None
        self.grey_scale = None
        self.IM = None
        self.glimpse = None
        self.focus = None
        self.init_scale = None
        
        #SPECIFYING GLIMPSE SIZE
        self.glimpse_size_h, self.glimpse_size_w = glimpse_size
        self.level_depth, self.level_scale = levels                 
        
        #INTITIALIZING STATE [0, 0]
        self.initial_state = np.zeros((1,2), dtype="float32")  
        
        #HISTORY
        self.enable_history = enable_history
        self.glimpse_history = []
   
        #DEBUG
        self.debug = debug
        self.optimized = optimized
    
    def reset(self):
        self.swap_channel_major = False
        self.grey_scale = False
        self.IM = None
        self.focus = self.initial_state                 # [Batch, 2 (x,y)]
        self.init_scale = 0
        self.glimpse = None
        del self.glimpse_history[:]        
        
    
    def load_batch(self, IM, compute_initial_state = False):
        """
        Parameters
        ----------
        images       as ([B, C, H, W])   
        """
        self.reset()
        
        if IM.ndim > 4:
            print("Tensor not valid. Got: ", IM.shape,  ", but expected [Batch, Channels, Height, Width]")
            return
        
        if IM.ndim == 4 and IM.shape[1] > 4:             # IM [Batch, Channels, Height, Width]
            #CHECK IF CHANNEL-MAJOR
            self.IM = IM.swapaxes(1,3).swapaxes(2,3)
            self.swap_channel_major = True
        else:
            #ADD DIMENSION FOR GREY SCALE
            if IM.ndim == 3:
                self.IM = IM.reshape(IM.shape[0], 1, IM.shape[1],IM.shape[2])
                self.grey_scale = True
            else:
                if IM.ndim == 4 and IM.shape[1] == 1:
                    self.grey_scale = True
                self.IM = IM
        
        if self.debug:
            print("IM.ndim: ", IM.ndim)
            print(self.IM.shape)
            print("self.swap_channel_major", self.swap_channel_major)
            print("self.grey_scale", self.grey_scale) 
        
        if compute_initial_state:
            self.step(self.initial_state.repeat(IM.shape[0], axis=0))            
        
    
    def step(self, position, init_scale=None):
        """
        Performs one step in the loaded episode.
        
        Parameters
        ----------
        position   as ([Batch, 2 (x, y)]) x,y e [-1, +1]
        init_scale is ([Batch, 1]) initial scale of the glimpse (for levels=2 and scale=0, 2^1, and 2^2 is returned)
        
        Return
        ------
        state  as ([Batch, levels, Glimpse_height, Glimpse_width])
        reward as (scalar)
        done   as (Boolean)
        None   as (None) aka Placeholder        
        """
        if self.IM is None:
           print("No image loaded. Did you call load_image()?")
           return
        
        #UPDATE STATE
        if position.shape[0] == self.IM.shape[0]: 
            if position.dtype == "float32":
                self.focus = position                         # [Batch, 2 (x, y)]
                self.init_scale = init_scale
            else:
                print("DTYPE mismatch. Expected: float32 but was:", position.dtype)
                return
        else:
            print("Location size mismatch. Expected:", self.IM.shape[0], ", but was:", position.shape[0])
            return
              
        if self.debug:
            print("(DEBUG) new state: ", position)
        
        #COMPUTE STATE
        if self.optimized == False:
            self.glimpse = focal.comp_focal_pyramide_opt(self.IM, self.focus, self.glimpse_size_h, self.glimpse_size_w, self.level_depth, self.level_scale, self.init_scale)
        else:
            if self.glimpse_size_h != self.glimpse_size_w and self.level_scale != 2:
                raise NotImplementedError("Only level_scale=2 and quadratic glimpses are optimized!: Got: glimpse_size_h", self.glimpse_size_h ,"glimpse_size_w",self.glimpse_size_w, "and level_scale", self.level_scale)
            self.glimpse = focal.comp_focal_pyramide_opt(self.IM, self.focus, self.glimpse_size_h, self.level_depth, self.init_scale)
        
        
        #COMPUTE REWARD (Scale 2 = -0.2, Scale 1 = -0.1, Scale 0 = 0)
        reward = (-1 * init_scale)
        
        #IS FINISHED
        done = False
        
        #Append to history
        if self.enable_history:
            self.glimpse_history.append((self.IM[0:9], self.glimpse[0:9], self.focus[0:9], self.init_scale[0:9] if self.init_scale is not None else None))
        
        return self.render(), reward, done, None
 
 
        
    def render(self, for_plot = False):
        """
        Returns the current state of the sensor
        """
        if self.IM is None:
            print("No image loaded. Did you call load_image()?")
            return
        
        if self.grey_scale:
            return self.glimpse.reshape(self.glimpse.shape[0], self.glimpse.shape[1], self.glimpse.shape[3], self.glimpse.shape[4])
        else:
            if for_plot or self.swap_channel_major:
                return self.glimpse.swapaxes(2,4).swapaxes(2,3)
            else:
                return self.glimpse
        
            
    def plot(self, num_images=-1):
        """
        Plots the current state of the sensor with matplotlib
        """
        if self.IM is None:
            print("No image loaded. Did you call load_image()?")
            return
        
        if self.glimpse is None:
            print("No glimpse computed.")
            return
        
        if num_images == -1:
            num_images = self.glimpse.shape[0]
        
        num_levels = self.glimpse.shape[1]
        if num_images > 9:
            print("I can only plot up to 9 images. But", num_images, "were given.")
            num_images = 9
            
        if num_levels > 9:
            print("I can only plot up to 9 levels. But", num_levels, "were given.")
            num_levels = 9
        
        for b in range(num_images):
            for i in range(num_levels):
                plt.subplot(33 * 10 + (i) + 1)
                if self.grey_scale:
                    plt.imshow(self.render(True)[b][i].astype("float32"), cmap="gray")
                else:
                    plt.imshow(self.render(True)[b][i].astype("float32"))
                plt.axis('off')
            plt.show()
            
    def plot_history(self, num_images=-1, num_levels=-1):
        """
        Plots the historc state of the sensor in with matplotlib
        """
        if self.IM is None:
            print("No image loaded. Did you call load_image()?")
            return
        
        if self.glimpse is None:
            print("No glimpse computed.")
            return
                
        if self.enable_history == False:
            print("History is not enabled. Have you initialized with enable_history=True?")
        
        if num_images == -1:
            num_images = self.glimpse.shape[0]
        
        if num_levels == -1:
            num_levels = self.glimpse.shape[1]
        
        num_history = len(self.glimpse_history)            
        
        if num_images > 9:
            print("I can only plot up to 9 images. But", num_images, "were given.")
            num_images = 9
            
        if num_levels > 9:
            print("I can only plot up to 9 levels. But", num_levels, "were given.")
            num_levels = 9
        if num_history > 9:
            print("I can only plot up to 9 steps in history. But", num_history, "were given.")
            num_history = 9
        
        # Backup Glimpse
        glimpse_backup = self.glimpse
        for b in range(num_images):
            for h in range(num_history):
                IM, glimpse, focus, init_scale = self.glimpse_history[h]
                self.glimpse = glimpse
                for i in range(num_levels):
                    plt.subplot(num_levels, num_history, (i * num_history)  + (1 + h) ) # 1,4,2,5,3,6,
                    if self.grey_scale:
                        plt.imshow(self.render(True)[b][i].astype("float32"), cmap="gray")
                    else:
                        plt.imshow(self.render(True)[b][i].astype("float32"))
                    plt.axis('off')
            plt.show()
        
        # Restore Glimpse
        self.glimpse = glimpse_backup
        
        
    def plot_history_full_images(self, num_images=-1, num_levels=-1, scale=2):
        """
        Plots the historc state of the sensor  with matplotlib
        """
        if self.IM is None:
            print("No image loaded. Did you call load_image()?")
            return
        
        if self.glimpse is None:
            print("No glimpse computed.")
            return
                
        if self.enable_history == False:
            print("History is not enabled. Have you initialized with enable_history=True?")
        
        if num_images == -1:
            num_images = self.glimpse.shape[0]
        
        if num_levels == -1:
            num_levels = self.glimpse.shape[1]
        
        num_history = len(self.glimpse_history)            
        
        if num_images > 9:
            print("I can only plot up to 9 images. But", num_images, "were given.")
            num_images = 9
            
        if num_levels > 9:
            print("I can only plot up to 9 levels. But", num_levels, "were given.")
            num_levels = 9
        if num_history > 9:
            print("I can only plot up to 9 steps in history. But", num_history, "were given.")
            num_history = 9
            
        patch_h = self.glimpse.shape[3]
        patch_w = self.glimpse.shape[4]   
        
        im_h = self.IM.shape[2]
        im_w = self.IM.shape[3]
        
        for b in range(num_images):
            for h in range(num_history):
                plt.subplot(1, num_history, (h + 1))
                IM_h, glimpse_h, focus_h, init_scale_h = self.glimpse_history[h]
                IM = IM_h[b,:,:,:].copy()
                        
                for i in range(init_scale_h[b][0], num_levels + init_scale_h[b][0], 1):
               
                    pos_xl = int(((focus_h[b, 0] + 1 ) * im_h/2) - (0.5 * patch_h * scale**i))
                    pos_yu = int(((focus_h[b, 1] + 1 ) * im_w/2) - (0.5 * patch_w * scale**i))
                    pos_xr = int(((focus_h[b, 0] + 1 ) * im_h/2) + (0.5 * patch_h * scale**i))
                    pos_yb = int(((focus_h[b, 1] + 1 ) * im_w/2) + (0.5 * patch_w * scale**i))
                    
                    if pos_xl < 0:
                        pos_xl = 0
                    if pos_yu < 0:
                        pos_yu = 0
                    if pos_xr >= im_h:
                        pos_xr = im_h - 1
                    if pos_yb >= im_w:
                        pos_yb = im_w - 1
                    if pos_xr < 0:
                        pos_xr = 0
                    if pos_yb < 0:
                        pos_yb = 0
                    if pos_xl >= im_h:
                        pos_xl = im_h - 1
                    if pos_yu >= im_w:
                        pos_yu = im_w - 1
                  
                    IM[:, pos_xl, pos_yu:pos_yb] = 1.
                    IM[:, pos_xr, pos_yu:pos_yb] = 1.
                    IM[:, pos_xl:pos_xr, pos_yu] = 1.
                    IM[:, pos_xl:pos_xr, pos_yb] = 1.
                    IM[:, pos_xr, pos_yb] = 1.
                
                if self.grey_scale:
                    IM = IM.reshape(IM.shape[1], IM.shape[2])
                   
                else:
                    IM = IM.swapaxes(0,2).swapaxes(0,1)
                if self.grey_scale:
                    plt.imshow(IM.astype("float32"), cmap="gray")
                else:
                    plt.imshow(IM.astype("float32"))
                plt.axis('off')
            plt.show()
              
    
if __name__ == "__main__":
    
    # GENERATE DUMMY IMAGE (batch_size = 3)
    IM = np.random.rand(3, 3, 30,30)
    
    
    glimpse_sensor = GlimpseSensor(images, targets, batch_size=args.batch_size, loop=True, random=True, glimpse_size=(args.glimpse_size, args.glimpse_size), levels=(args.glimpse_level, 2), debug=False, optimized=True)                
  
    # Re-positioning the sensor
    pyramide, _, __, ___  = glimpse_sensor.step(new_location)
    
    # Skip to next episode (aka load new mini-batch of images)
    glimpse_sensor.next_episode()
   
    # INSTANTIATE GlimpseSensor (10 levels, doubled scaled, size 20 x 20)
    env = GlimpseSensor((20,20), (6, 2), debug = False, optimized=False)       
    env.load_image(IM, compute_initial_state = True)   
    
    # Plot the state
    env.plot(num_images=1)
    
    res = env.step(np.array([[10,10],[10,10],[10,10]], "int32"))

    # Re-positioning the sensor (totally outside)
    res = env.step(np.array([[1000,1000],
                             [1000,1000],
                             [1000,1000]], "int32"))    
    
    # Plot the state
    env.plot()
    
    # Reset the state
    res = env.reset()
    
    # Load environment and sensor
    IM = (np.random.rand(1000, 3, 100,100) * 255).astype("int32")
    env_new = GlimpseSensor((20,20), (2, 2), debug = False, enable_history=True, optimized=True)       
    environment_new = Environment(env_new, IM, batch_size = 1000, loop=True, random=True, debug=False)
    env_old = GlimpseSensor((20,20), (3, 2), debug = False, optimized=False)     
    environment_old = Environment(env_old, IM, batch_size = 1000, loop=True, random=True, debug=False)
    
    # Go To First state
    environment_new.next()
    environment_old.next()
    
    # Take step
    glimpse1, _, __, ___ = env_new.step(np.zeros((1000,2)).astype("float32"))   
    glimpse1, _, __, ___ = env_new.step(np.zeros((1000,2)).astype("float32"))   
    glimpse1, _, __, ___ = env_new.step(np.zeros((1000,2)).astype("float32"))   
    glimpse1, _, __, ___ = env_new.step(np.zeros((1000,2)).astype("float32"))   
    
    env_new.plot_history_full_images(num_images=1)
    
    glimpse2, _, __, ___ = env_old.step(np.ones((1000,2)).astype("float32"))
        

    
    env_new.plot()
    env_old.plot()
    
    # Plot the state
    # Test throughput
    #%timeit environment.next()
    #%timeit env.step(np.arange(40 * 2).reshape(40,2).astype("int32"))  
    
    
    #IM = mpimg.imread("/home/matthias/Desktop/workspace/code-samples/Visual-Attention/ikosaeder.png")
    #IM = np.expand_dims(IM, axis=0)
    #IM = np.repeat(IM, 40, axis=0)
