from __future__ import print_function

from skimage.feature import daisy
from skimage import color

import numpy as np
import math
import scipy.misc

import os
import yaml
cfg=yaml.safe_load(open('core/config.yaml'))
type_ex='daisy'
n_slice    = cfg[type_ex]['n_slice']
n_orient   = cfg[type_ex]['n_orient']
step       = cfg[type_ex]['step']
radius     = cfg[type_ex]['radius']
rings      = cfg[type_ex]['rings']
histograms = cfg[type_ex]['histograms']
h_type     = cfg[type_ex]['h_type']

class Daisy(object):
    def __init__(self, type=h_type, n_slice=n_slice, normalize=True):
        self.type=h_type
        self.n_slice=n_slice
        self.normalize=normalize
        self.R = (rings * histograms + 1) * n_orient
        if h_type=='global':
            self.dimension = self.R
        else:
            self.dimension = self.R*self.n_slice*self.n_slice
    def feature(self, input):
        '''
    
        arguments
            input    : a path to a image or a numpy.ndarray
            type     : 'global' means count the feature for whole image
                    'region' means count the feature for regions in images, then concatanate all of them
            n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
            normalize: normalize output feature
    
        return
            type == 'global'
            a numpy array with size R
            type == 'region'
            a numpy array with size n_slice * n_slice * R
    
            #R = (rings * features + 1) * n_orient#
        '''
        if isinstance(input, np.ndarray):  # examinate input type
            img = input.copy()
        else:
            img = scipy.misc.imread(input, mode='RGB')
        height, width, channel = img.shape
    
        P = math.ceil((height - radius*2) / step) 
        Q = math.ceil((width - radius*2) / step)
        assert P > 0 and Q > 0, "input image size need to pass this check"
    
        if self.type == 'global':
            hist = self._daisy(img)
    
        elif self.type == 'region':
            hist = np.zeros((self.n_slice, self.n_slice, self.R))
            h_silce = np.around(np.linspace(0, height, self.n_slice+1, endpoint=True)).astype(int)
            w_slice = np.around(np.linspace(0, width, self.n_slice+1, endpoint=True)).astype(int)
    
            for hs in range(len(h_silce)-1):
                for ws in range(len(w_slice)-1):
                    img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
                    hist[hs][ws] = self._daisy(img_r)
    
        if self.normalize:
            hist /= np.sum(hist)
    
        return hist.flatten()
  
  
    def _daisy(self, img, normalize=True):
        image = color.rgb2gray(img)
        descs = daisy(image, step=step, radius=radius, rings=rings, histograms=histograms, orientations=n_orient)
        descs = descs.reshape(-1, self.R)  # shape=(N, R)
        hist  = np.mean(descs, axis=0)  # shape=(R,)  
        return hist

if __name__ == "__main__":
    _daisy = Daisy( type='region', n_slice=2)
    input = '../../data/style/0_0_001.png'
    # test normalize
    # hist = color.feature(input, type='global')
    hist = _daisy.feature(input)
    print(hist.__len__())
    input = '../../data/style/0_0_002.png'
    # test normalize
    # hist2 = color.feature(input, type='global')
    hist2 = _daisy.feature(input)
    # print(hist2)
    print(np.sum((hist - hist2) ** 2))