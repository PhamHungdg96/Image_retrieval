from __future__ import print_function

from skimage.filters import gabor_kernel
from skimage import color
from scipy import ndimage as ndi

import multiprocessing
import numpy as np
import os
import cv2

import yaml
cfg=yaml.safe_load(open('core/config.yaml'))
type_ex='gabor'
theta     = cfg[type_ex]['theta']
frequency = tuple(float(i) for i in cfg[type_ex]['frequency'].split(' '))
sigma     = tuple(float(i) for i in cfg[type_ex]['sigma'].split(' '))
bandwidth = tuple(float(i) for i in cfg[type_ex]['bandwidth'].split(' '))

n_slice  = cfg[type_ex]['n_slice']
h_type   = cfg[type_ex]['h_type']

def make_gabor_kernel(theta, frequency, sigma, bandwidth):
    kernels = []
    for t in range(theta):
        t = t / float(theta) * np.pi
        for f in frequency:
            if sigma:
                for s in sigma:
                    kernel = gabor_kernel(f, theta=t, sigma_x=s, sigma_y=s)
                    kernels.append(kernel)
            if bandwidth:
                for b in bandwidth:
                    kernel = gabor_kernel(f, theta=t, bandwidth=b)
                    kernels.append(kernel)
    return kernels

gabor_kernels = make_gabor_kernel(theta, frequency, sigma, bandwidth)
if sigma and not bandwidth:
    assert len(gabor_kernels) == theta * len(frequency) * len(sigma), "kernel nums error in make_gabor_kernel()"
elif not sigma and bandwidth:
    assert len(gabor_kernels) == theta * len(frequency) * len(bandwidth), "kernel nums error in make_gabor_kernel()"
elif sigma and bandwidth:
    assert len(gabor_kernels) == theta * len(frequency) * (len(sigma) + len(bandwidth)), "kernel nums error in make_gabor_kernel()"
elif not sigma and not bandwidth:
    assert len(gabor_kernels) == theta * len(frequency), "kernel nums error in make_gabor_kernel()"

class Gabor(object):
    ''' count img feature
    
        arguments
            input    : a path to a image or a numpy.ndarray
            type     : 'global' means count the feature for whole image
                    'region' means count the feature for regions in images, then concatanate all of them
            n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
            normalize: normalize output feature
    
        return
            type == 'global'
            a numpy array with size len(gabor_kernels)
            type == 'region'
            a numpy array with size len(gabor_kernels) * n_slice * n_slice
        '''
    def __init__(self,type=h_type, n_slice=n_slice, normalize=True):
        self.type=h_type
        self.n_slice=n_slice
        self.normalize=normalize
        self.dimension= len(gabor_kernels) if  self.type == 'global' else len(gabor_kernels)*self.n_slice**2
    def feature(self, input):
        if isinstance(input, np.ndarray):  # examinate input type
            img = input.copy()
        else:
            img = cv2.imread(input)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
    
        if self.type == 'global':
            hist = self._gabor(img, kernels=gabor_kernels)
    
        elif self.type == 'region':
            hist = np.zeros((self.n_slice, self.n_slice, len(gabor_kernels)))
            h_silce = np.around(np.linspace(0, height, self.n_slice+1, endpoint=True)).astype(int)
            w_slice = np.around(np.linspace(0, width, self.n_slice+1, endpoint=True)).astype(int)
        
            for hs in range(len(h_silce)-1):
                for ws in range(len(w_slice)-1):
                    img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
                    hist[hs][ws] = self._gabor(img_r, kernels=gabor_kernels)
    
        if self.normalize:
            hist /= np.sum(hist)
    
        return hist.flatten()[:self.dimension]
  
  
    def _feats(self, image, kernel):
        '''
        arguments
            image : ndarray of the image
            kernel: a gabor kernel
        return
            a ndarray whose shape is (2, )
        '''
        feats = np.zeros(2, dtype=np.double)
        filtered = ndi.convolve(image, np.real(kernel), mode='wrap')
        feats[0] = filtered.mean()
        feats[1] = filtered.var()
        return feats
  
  
    def _power(self, image, kernel):
        '''
        arguments
            image : ndarray of the image
            kernel: a gabor kernel
        return
            a ndarray whose shape is (2, )
        '''
        image = (image - image.mean()) / image.std()  # Normalize images for better comparison.
        f_img = np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                    ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
        feats = np.zeros(2, dtype=np.double)
        feats[0] = f_img.mean()
        feats[1] = f_img.var()
        return feats
  
  
    def _gabor(self, image, kernels=make_gabor_kernel(theta, frequency, sigma, bandwidth), normalize=True):
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
        img = color.rgb2gray(image)
    
        results = []
        feat_fn = self._power
        for kernel in kernels:
            results.append(pool.apply_async(self._worker, (img, kernel, feat_fn)))
        pool.close()
        pool.join()
        
        hist = np.array([res.get() for res in results])
    
        return hist.T.flatten()
  
  
    def _worker(self, img, kernel, feat_fn):
        try:
            ret = feat_fn(img, kernel)
        except:
            print("return zero")
            ret = np.zeros(2)
        return ret


if __name__ == "__main__":
    gabor = Gabor(type='global', n_slice=2)
    input = '../../data/style/0_0_001.png'
    # test normalize
    # hist = color.feature(input, type='global')
    hist = gabor.feature(input)
    print(hist.__len__())
    input = '../../data/style/0_0_002.png'
    # test normalize
    # hist2 = color.feature(input, type='global')
    hist2 = gabor.feature(input)
    # print(hist2)
    print(np.sum((hist - hist2) ** 2))
