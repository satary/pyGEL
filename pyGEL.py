# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:30:56 2016

@author: armeev
"""
import numpy as np
from general_functions import box_transform_1channel, Distribution, squarte_box

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
        
        
#    def creation_lane(self,points):
#        for         
#        lane={}
        
        

class Lane(object):
    def __init__(self,data=None, point=None, data_mass=None, data_text=None):
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
        else:
            self.data= np.loadtxt(data_text)

class Fit(object):
    '''
    fitting curve and optimizition 
    select distribution lorenzian or gaussian or weibull
    '''
    def __init__(self,line, metod = "lorenzian", sigma=2,thres_min=0.1,thres_peak=0.1,min_dist=1, maxiter =2):
        '''
        sigma - Standard deviation for Gaussian kernel.
        thres_peak -  Normalized threshold. Only the peaks with amplitude higher than the threshold will be detected.
        thres_min - For local minim. Normalized threshold. Only the peaks with amplitude higher than the threshold will be detected.
        min_dist -  Minimum distance between each detected peak. The peak with the highest amplitude is preferred to satisfy this constraint.
        maxinter - Maximum number of iterations to perform.
        '''
        self.data=line
        if metod=="lorenzian" or metod=="gaussian":
            if metod=="lorenzian":
                '''
                distribution Lorenzian
                
                y = area * 1/pi *(0.5*width)/((x-center)**2 +(0.5*width)**2)
        
                area - the area under curve
                center - position one peak
                width - width distibution 
                
                return optimized widht peaks, area peaks(probability break chain DNA) and centers peaks
                '''
                self.lorenzian = Distribution(self.data)
                self.data=self.lorenzian.optim_lorenzian(sigma,thres_min,thres_peak,min_dist, maxiter)
            else:
                '''
                distribution Gaussa 
                
                y= area * 1/(width*sqrt(2*pi)) * exp(-(x-center)**2/(2*width**2))
                
                area - the area under curve
                center - position one peak
                width - width distibution
                
                return optimized widht peaks, area peaks(probability break chain DNA) and centers peaks
                '''
                self.gaussian = Distribution(self.data)
                self.data = self.gaussian.optim_gaussian(sigma,thres_min,thres_peak,min_dis, maxiter)
        else:
            '''
            distribution Weibull            
            '''
            self.weibull = Distribution(self.data)
            self.data = self.weibull.optim_weibull(sigma,thres_min,thres_peak,min_dist, maxiter)

#points=np.array([[ 80,  20],
#                 [500, 182],
#                 [ 98, 440],
#                 [400,460]])
#                 
#a=Gel('running_dog.jpg')
##a.rgb2grayscale()
#lol=a.box_transform(points)
#a.show()
