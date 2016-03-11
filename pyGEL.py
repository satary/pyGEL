# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:30:56 2016

@author: armeev
"""
import numpy as np
from general_functions import box_transform_1channel, Fit, squarte_box

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
        
    def get_lane(self,point):
        return Lane(self.image,point)
        

        

class Lane(object):
    def __init__(self,data=None, point=None, data_mass=None):
        if not(data is None):
            '''
            given image with points 
            '''
            self.data=data
            if not(point is None):
                if point[:,0].size==2:
                    self.data = squarte_box(point,self.data)
                    self.data = np.average(self.data,1)[:,1]
                    self.data = 256 -self. data
                elif point[:,0].size==4:
                    self.data = box_transform_1channel(point,self.data)
                    self.data = np.average(self.data,1)[:,1]
                    self.data = 256 -self. data
                else:
                    raise ValueError('points should be 2 or 4')
            else:
                self.data=mpimg.imread(data)
                self.data = np.average(self.data,1)[:,1]
                self.data = 256 -self. data
        elif not(data_mass is None):
             self.data = data_mass

     
     

#points=np.array([[ 80,  20],
#                 [500, 182],
#                 [ 98, 440],
#                 [400,460]])
#                 
#a=Gel('running_dog.jpg')
##a.rgb2grayscale()
#lol=a.box_transform(points)
#a.show()
