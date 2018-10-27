#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:03:42 2018

@author: rosalyn
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image

def psnr(A, B):
    return 10*np.log(255*255.0/(((A.astype(np.float)-B)**2).mean()))/np.log(10)

def double2uint8(I, ratio=1.0):
    return np.clip(np.round(I*ratio), 0, 255).astype(np.uint8)

def make_kernel(f):
    kernel = np.zeros((2*f+1, 2*f+1))
    for d in range(1, f+1):
        kernel[f-d:f+d+1, f-d:f+d+1] += (1.0/((2*d+1)**2))
    return kernel/kernel.sum()

def NLmeansfilter(I, h_=10, templateWindowSize=5,  searchWindowSize=11):
    f = int(templateWindowSize/2)
    t = int(searchWindowSize/2)
    height, width = I.shape[:2]
    padLength = t+f
    I2 = np.pad(I, int(padLength), 'symmetric')
    kernel = make_kernel(int(f))
    h = (h_**2)
    I_ = I2[int(padLength-f):int(padLength+f+height), int(padLength-f):int(padLength+f+width)]

    average = np.zeros(I.shape)
    sweight = np.zeros(I.shape)
    wmax =  np.zeros(I.shape)
    for i in range(-t, t+1):
        for j in range(-t, t+1):
            if i==0 and j==0:
                continue
            I2_ = I2[int(padLength+i-f):int(padLength+i+f+height), int(padLength+j-f):int(padLength+j+f+width)]
            w = np.exp(-cv2.filter2D((I2_ - I_)**2, -1, kernel)/h)[f:f+height, f:f+width]
            sweight += w
            wmax = np.maximum(wmax, w)
            average += (w*I2_[f:f+height, f:f+width])
    return (average+wmax*I)/(sweight+wmax)
def im_denoise(im_dir='image_hs',sigma=25,de_im_dir='de_img_hs'):
    if not os.path.exists(de_im_dir):
        os.makedirs(de_im_dir)
    im_names=os.listdir(im_dir)
    for im in im_names:
        img=cv2.imread(os.path.join(im_dir,im),0)
        img_ = double2uint8(NLmeansfilter(img.astype(np.float), sigma, 5, 11))
        de_im=Image.fromarray(img_).convert('L')
        de_im.save(os.path.join(de_im_dir,im))
        
        

if __name__ == '__main__':
    

    
    im_denoise()


