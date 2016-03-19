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




class Distribution(object):
    def __init__(self, line):
        self.line=line
        self.line_reflected = max(self.line) - self.line
        self.line_new=self.line[self.position_min_peak()[0]:self.position_min_peak()[-1]]
        self.x_axis = np.arange(self.line_new.size)
        
        
    def position_min_peak(self, sigma=2, thres=0.1, min_dist=1):
        '''
        sigma - Standard deviation for Gaussian kernel.
        thres -  Normalized threshold. Only the peaks with amplitude higher than the threshold will be detected.
        min_dist -  Minimum distance between each detected peak. The peak with the highest amplitude is preferred to satisfy this constraint. 
        '''
        line_reflected = ndimage.filters.gaussian_filter(self.line_reflected,sigma=sigma)
        index_min = peakutils.peak.indexes(line_reflected, thres=thres, min_dist=min_dist)
        return index_min

        
        
    def index_peak(self, sigma=2,thres=0.1,min_dist=1):
        '''
        sigma - Standard deviation for Gaussian kernel.
        thres -  Normalized threshold. Only the peaks with amplitude higher than the threshold will be detected.
        min_dist -  Minimum distance between each detected peak. The peak with the highest amplitude is preferred to satisfy this constraint. 
        '''
        line = ndimage.filters.gaussian_filter(self.line_new,sigma=sigma)
        indexes = peakutils.peak.indexes(line, thres=thres, min_dist=min_dist)
        return indexes
        
    def center_peak(self):
        '''
        position peak
        '''
        center=self.x_axis[self.index_peak()]
        return center
        
    def height_peak(self):
        
        height=self.line[self.index_peak()]
        return height
    

        
    def one_peak(self,i):
        '''
        i-number peak
        '''
        return self.line[self.position_min_peak()[i]:self.position_min_peak()[i+1]]        
        
    def area_peak(self,i):
        '''
        area one peak
        i- number peak
        '''
        a = self.one_peak(i)[:-1]
        b = self.one_peak(i)[1:]
        area = 0.5*np.sum(a+b)
        return area 


    '''
    distribution Lorenzian
    '''        
        
    def width_and_area_peak_lorn(self):
        '''
        appoximate width and coefficient(area) distribution Lorenzian
        '''
        width = np.zeros(self.position_min_peak().size-1)
        area = np.zeros(self.position_min_peak().size-1)
        for i in np.arange(self.position_min_peak().size-1):
            width[i] = (2/np.pi)*(self.area_peak(i)/self.height_peak()[i])
            '''
            w = 2/(pi * y),  y = h/S
            '''
            area[i] = self.area_peak(i) 
        return width , area
        
    def guess_lorn(self):
        '''
        parametrs distribution Lorenzian (area, center, width)
        '''
        guess = np.hstack((self.width_and_area_peak_lorn()[0],self.width_and_area_peak_lorn()[1],self.center_peak()))
        return guess
        
    
    def lorenzian(self,height,width,x,center):
        '''
        distribution Lorenza (area, center, width)
        
        y = area * 1/pi *(0.5*width)/((x-center)**2 +(0.5*width)**2)
        
        area - the area under curve
        center - position one peak
        width - width distibution        
        '''
        lorn = height*(1.0/np.pi)*(0.5*width)/((x - center)**2+(0.5*width)**2)
        return lorn
        
    def sum_lorenzian(self,args,x):
        '''
        sum lorenzians all peaks
        '''
        args = args.reshape(3,-1)
        height=args[1]
        width=args[0]
        center=args[2]
        lorn_sum=np.zeros(x.size)
        for i in np.arange(center.size):
            lorn_sum+= height[i]*(1.0/np.pi)*(0.5*width[i])/((x - center[i])**2+(0.5*width[i])**2)
        return lorn_sum 
        
    def sum_difference_square_lorenzian(self, args, x, y):
        return np.sum((self.sum_lorenzian(args,x)-y)**2)     
 
 
 
    '''
    distribution Gaussian
    '''
       
    def width_and_area_peak_gaus(self):
        '''
        appoximate width and coefficient(area) distribution Gaussian
        '''
        width = np.zeros(self.position_min_peak().size-1)
        area = np.zeros(self.position_min_peak().size-1)
        for i in np.arange(self.position_min_peak().size-1):
            width[i] = (1/np.sqrt(2*np.pi))*(self.area_peak(i)/self.height_peak()[i])
            '''
            w = 1/(sqpr(2*pi) * y),  y = h/S
            '''
            area[i] = self.area_peak(i) 
        return width , area
        
    def guess_gaus(self):
        '''
        parametrs distribution Gaussian (area, center, width)
        '''
        guess = np.hstack((self.width_and_area_peak_gaus()[0],self.width_and_area_peak_gaus()[1],self.center_peak()))
        return guess    
            
    
    def gaussian(self,height,width,x,center):
        '''
        distribution Gaussa (area, center, width)
        
        y= area * 1/(width*sqrt(2*pi)) * exp(-(x-center)**2/(2*width**2))
        
        area - the area under curve
        center - position one peak
        width - width distibution
        '''
        gaus = height*1.0/(width*np.sqrt(2*np.pi))*np.exp(-(x - center)**2/(2*width**2))
        return gaus
        
    def sum_gaussian(self, args,x):
        '''
        sum gaussian all peak
        '''
        args=args.reshape(3,-1)
        center=args[2]
        height=args[1]
        width=args[0]
        gaus_sum=np.zeros(x.size)
        for i in np.arange(center.size):
            gaus_sum+= height[i]*1.0/(width[i]*np.sqrt(2*np.pi))*np.exp(-(x - center[i])**2/(2*width[i]**2))
        return gaus_sum
        
    def sum_difference_square_gaussian(self, args, x, y):
        return np.sum(( self.sum_gaussian(args,x)-y)**2) 
    
    '''
    disrtibution Weibull
    '''
    def width_and_area_peak_weibull(self):
        '''
        appoximate width and coefficient(area) distribution Weibull
        '''
        width = np.zeros(self.position_min_peak().size-1)
        area = np.zeros(self.position_min_peak().size-1)
        for i in np.arange(self.position_min_peak().size-1):
            width[i] = (2/np.sqrt(np.pi))*(self.area_peak(i)/self.height_peak()[i])
            '''
            w = 2/(sqpr(pi) * y),  y = h/S
            '''
            area[i] = self.area_peak(i) 
        return width , area
        
    def guess_weibull(self):
        '''
        parametrs distribution Weibull (area, center, width)
        '''
        guess = np.hstack((self.width_and_area_peak_weibull()[0],self.width_and_area_peak_weibull()[1],self.center_peak()))
        return guess   
        
    def weibull(self,height,width,x,center):
        weibull= height*(2/(width*np.sqrt(np.pi)))*np.exp(-0.5*((x-center)/width)**2)
        return weibull
    
    def sum_weibull(self,args,x):
        args=args.reshape(3,-1)
        center=args[2]
        height=args[1]
        width=args[0]
        weib_sum=np.zeros(x.size)
        for i in np.arange(center.size):
            weib_sum+= height[i]*(2/(width[i]*np.sqrt(np.pi)))*np.exp(-0.5*((x-center[i])/width[i])**2)
        return weib_sum 
        
    def sum_difference_square_weibull(self,args, x, y):
        return np.sum(( self.sum_weibull(args,x)-y)**2) 
       
    def optim_lorenzian(self):
        '''
        optimization Lorenzian
        '''
        optim = optimize.fmin_powell(self.sum_difference_square_lorenzian, self.guess_lorn() , args=(self.x_axis, self.line_new), maxiter = 2)
        return optim
        
    def optim_gaussian(self):
        '''
        optimization Gaussian
        '''
        optim = optimize.fmin_powell(self.sum_difference_square_gaussian, self.guess_gaus() , args=(self.x_axis, self.line_new), maxiter = 2)
        return optim
    
    def optim_weibull(self):
        '''
        optimization Weibull
        '''
        optim = optimize.fmin_powell(self.sum_difference_square_weibull, self.guess_weibull() , args=(self.x_axis, self.line_new), maxiter = 2)
        return optim
    
    def show_peak(self,arg):
#        arg = self.optim_lorenzian().reshape(2,-1)
        arg=arg.reshape(3,-1)
        height=arg[1]
        width=arg[0]
        center=arg[2]
        for i in np.arange(center.size):
            plt.plot(self.x_axis, self.weibull(height[i],width[i],self.x_axis,center[i]))
            plt.show()
    
    def show(self,arg):
        plt.plot(self.line_new)
        plt.plot(self.x_axis,self.sum_weibull(arg, self.x_axis))
        plt.show()
        
        
        
def ratio_height_width(arg):
    arg=arg.reshape(2,-1)
    c = np.zeros(arg[0].size)
    d = np.arange(arg[0].size) 
    for i in np.arange(arg[0].size):
        c[i]+= arg[0,i]/arg[1,i]
    plt.plot(d,c)
    plt.show()




#class Fit(object):
#    '''
#    fitting function 
#    '''
#    def __init__(self,data):
#        
#    
#    def optim_lorenzian(self):
#        '''
#        getting area lorenzian
#        '''
#        optim_lorn = optimize.fmin_powell(self.min_lorn, self.guess , args=(self.x, self.y, self.center), maxiter=10)
#        return optim_lorn
#        
#    def optim_gaussian(self):
#        '''
#        getting area gaussian
#        '''
#        optim_gaus = optimize.fmin_powell(self.min_gaus, self.guess , args=(self.x, self.y, self.center), maxiter=10)
#        return optim_gaus
#        
#    def optim_weibull(self):
#        '''
#        getting area weibull
#        '''
#        optim_weib = optimize.fmin_powell(self.min_weib, self.guess , args=(self.x, self.y, self.center), maxiter=10)
#        return optim_weib
#        
#    def show(self,b):
#        plt.plot(self.x,self.y)
#        plt.plot(self.x, self.weibull(b,self.x))
#        plt.show()
##
#f=misc.imread('footprint_b1.png')
#nuc = f[50:1190, 10:20]
#nuc = np.average(nuc,1)[:,1]
#nuc = 256 - nuc
#a=Fit(nuc)
#b = a.optim_weibull() 
f=np.genfromtxt('20130116gel-BS-lane-3.txt')