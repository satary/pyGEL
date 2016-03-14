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


class Fit(object):
    '''
    fitting function 
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
        
    def gaussian(self, arg,x):
        shift=arg[0]
        args=arg[1:].reshape(3,-1)
        center=args[2]+shift
        height=args[0]
        width=args[1]
        guas_sum=np.zeros(x.size)
        for i in np.arange(center.size):
            gaus_sum+= height[i]*1.0/(width[i]*np.sqrt(2*np.pi))*np.exp(-(x - center[i])**2/(2*width[i]**2))
        return gaus_sum
        
    def weibull(self,arg,x):
        shift=arg[0]
        args=arg[1:].reshape(3,-1)
        center=args[2]+shift
        height=args[0]
        width=args[1]
        weib_sum=np.zeros(x.size)
        for i in np.arange(center.size):
            weib_sum+= height[i]*(0.5/width[i])*(((x-center[i])/width[i])**(-0.5))*np.exp(-1/0.5*((x-center[i])/width[i])**0.5)
        return weib_sum 
    
    def min_lorn(self, arg, x, y, centers):
        shift=arg[0]
        ar=arg[1:].reshape(3,-1)
        center=ar[2]+shift
        height=ar[0]
        width=ar[1]
        j=(width < 0) | (width > 0.05*height)
        i= height < 0 
        return np.sum(( self.lorenzian(arg,x)-y)**2) + 0.1*np.sum((centers-center)**2) + np.sum(height[i]**2) + np.sum(width[j]**2)
        
    def min_gaus(self, arg, x, y, centers):
        shift=arg[0]
        ar=arg[1:].reshape(3,-1)
        center=ar[2]+shift
        height=ar[0]
        width=ar[1]
        j=(width < 0) | (width > 0.05*height)
        i= height < 0 
        return np.sum(( self.gaussian(arg,x)-y)**2) + 0.1*np.sum((centers-center)**2) + np.sum(height[i]**2) + np.sum(width[j]**2)
        
    def min_weib(self, arg, x, y, centers):
        shift=arg[0]
        ar=arg[1:].reshape(3,-1)
        center=ar[2]+shift
        height=ar[0]
        width=ar[1]
        j=(width < 0) | (width > 0.05*height)
        i= height < 0 
        return np.sum(( self.weibull(arg,x)-y)**2) + 0.1*np.sum((centers-center)**2) + np.sum(height[i]**2) + np.sum(width[j]**2)
       
    def optim_lorenzian(self):
        '''
        getting area lorenzian
        '''
        optim_lorn = optimize.fmin_powell(self.min_lorn, self.guess , args=(self.x, self.y, self.center), maxiter=20)
        return optim_lorn
        
    def optim_gaussian(self):
        '''
        getting area gaussian
        '''
        optim_gaus = optimize.fmin_powell(self.min_gaus, self.guess , args=(self.x, self.y, self.center), maxiter=20)
        return optim_gaus
        
    def optim_weibull(self):
        '''
        getting area weibull
        '''
        optim_weib = optimize.fmin_powell(self.min_weib, self.guess , args=(self.x, self.y, self.center), maxiter=20)
        return optim_weib
        
    def show(self):
        plt.plot(self.x,self.y)
        plt.plot(self.x, weibull(self.optim_weibull(),self.x))
        plt.show()
#
f=misc.imread('footprint_b1.png')
nuc = f[50:1190, 10:20]
nuc = np.average(nuc,1)[:,1]
nuc = 256 - nuc
a=Fit(nuc)
#b = a.optim_weibull() 

#def weibull(arg,x):
#    shift=arg[0]
#    args=arg[1:].reshape(3,-1)
#    center=args[2]+shift
#    height=args[0]
#    width=args[1]
#    weib_sum=np.zeros(x.size)
#    for i in np.arange(center.size):
#        weib_sum+= height[i]*0.5/width[i]*((x-center[i])/width[i])**(-0.5)*np.exp(-1/0.5*((x-center[i])/width[i])**0.5)
#    return weib_sum
#
#plt.plot(a.x,weibull(b,a.x))
#plt.plot(a.x,a.y) 
#plt.show()  
#shift=b[0]
#o_nuc=b[1:].reshape(3,-1)
#c_nuc=o_nuc[2]+shift
#h_nuc=o_nuc[0]
#w_nuc=o_nuc[1]


#for i in np.arange(c_nuc.size):
#    plt.plot(a.x,np.zeros(a.x.size)+h_nuc[i]*0.5/w_nuc[i]*((a.x-c_nuc[i])/w_nuc[i])**0.5*np.exp(-1/0.5*((a.x-c_nuc[i])/w_nuc[i])**0.5))
    