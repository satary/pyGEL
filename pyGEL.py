# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:30:56 2016

@author: armeev
"""
import numpy as np
from general_functions import box_transform_1channel

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Gel(object):
    def __init__(self, path_to_image):
        self.image=mpimg.imread(path_to_image)
        self.color_depth=self.image.dtype
        self.nchannels=self.image.shape[-1]
    
    def rgb2grayscale(self):
        self.image=self.image.mean(2)
        
    def show(self):
        plt.imshow(self.image)
        plt.show()
        
    def box_transform(self,points,nx=0,ny=0):
        output=[]
        for channel in np.arange(self.nchannels):
            output.append(box_transform_1channel(points,self.image[:,:,channel],nx,ny))
        self.image=np.dstack(output).astype(self.color_depth)


points=np.array([[ 80,  20],
                 [500, 182],
                 [ 98, 440],
                 [400,460]])
                 
a=Gel('123.jpg')
#a.rgb2grayscale()
lol=a.box_transform(points)
#a.show()