# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:38:00 2016

@author: armeev
"""
import numpy as np
from scipy.interpolate import griddata
import peakutils
from scipy import ndimage, optimize, stats, misc

def build_grid(p,nx,ny):
    '''
    creation grid 4 points
    '''
    px=p[:,0]
    py=p[:,1]
    #creating matrixes of indexes for given nx and ny
    y_mat,x_mat=np.indices((ny+1,nx+1),dtype='float')
    x_mat/= nx
    y_mat/= ny
    #calculating coordinates of points in transform grid  
    #   p0 ----- p1
    #    |       |    
    #    |       |    
    #   p2-------p3
    #
    x_grid=px[0]+x_mat*(px[1]-px[0])-y_mat*(x_mat*(px[1]-px[0]-px[3]+px[2])-px[2]+px[0])
    y_grid=py[0]+y_mat*(py[2]-py[0])-x_mat*(y_mat*(py[2]-py[0]-py[3]+py[1])-py[1]+py[0])
    return x_grid, y_grid
    
def box_transform_1channel(points,input_img,nx=0,ny=0):
    '''
    creation new coords in image, 4 points
    '''
    if (nx == 0) or (ny == 0):
        nx= int(np.linalg.norm((points[0]-points[2]+points[1]-points[3])/2).round())
        ny= int(np.linalg.norm((points[0]-points[1]+points[2]-points[3])/2).round())  
    elif (nx < 0) or (ny < 0):
        raise ValueError('nx and ny should not be negative')
    x,y=build_grid(points,nx,ny)
    
    a,b=np.indices(input_img.shape)
    #this section reduces the initial image to simplify the task
    #rtol=100
    #xmin=0 if points[:,0].min()-rtol < 0 else points[:,0].min()-rtol
    #xmax=input_img.shape[0] if points[:,0].max()+rtol > input_img.shape[0] else points[:,0].max()+rtol
    #ymin=0 if points[:,1].min()-rtol < 0 else points[:,1].min()-rtol
    #ymax=input_img.shape[1] if points[:,1].max()+rtol > input_img.shape[1] else points[:,1].max()+rtol
    
    coords=np.dstack((b,a)).reshape(-1,2)
    return griddata(coords, input_img.flatten(), (x, y), method='linear')
    
def squarte_box(p,input_img):
    '''
    creation new coords in image, 2 points
    '''
    p1=p[0]
    p2=p[1]
    return input_img[p1[1]:p2[1],p1[0]:p2[0]]

#def track(p,lane):
#    

class Fit(object):
    '''
    fitting lorenzian 
    '''
    def __init__(self, line):
        self.line=line
        self.x = np.arange(self.line.size)
        self.y = ndimage.filters.gaussian_filter(self.line,sigma=2)
        self.indexes = peakutils.peak.indexes(self.y, thres=0.5, min_dist=5)
        self.center=self.x[self.indexes][::-1]
        self.height=self.y[self.indexes][::-1]
        self.width= np.zeros(self.height.size)+1
        self.guess=np.hstack((np.zeros(1),self.height,self.width,self.center))
        
        
    def lorenzian(self,arg,x):
        shift=arg[0]
        args=arg[1:].reshape(3,-1)
        center=args[2]+shift
        height=args[0]
        width=args[1]
        lorn_sum=np.zeros(x.size)
        for i in np.arange(center.size):
            lorn_sum+= height[i]*(1.0/np.pi)*(0.5*width[i])/((x - center[i])**2+(0.5*width[i])**2)
        return lorn_sum      
    
    def min_lorn(self, arg, x, y, centers):
        shift=arg[0]
        ar=arg[1:].reshape(3,-1)
        center=ar[2]+shift
        height=ar[0]
        width=ar[1]
        j=(width < 0) | (width > 0.05*height)
        i= height < 0 
        return np.sum(( self.lorenzian(arg,x)-y)**2) + 0.1*np.sum((centers-center)**2) + np.sum(height[i]**2) + np.sum(width[j]**2)
       
    def optim(self):
        '''
        getting area lorenzian
        '''
        optim = optimize.fmin_powell(self.min_lorn, self.guess , args=(self.x, self.y, self.center), maxiter=20)
        return optim
     