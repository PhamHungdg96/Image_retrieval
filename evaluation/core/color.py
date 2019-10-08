from __future__ import print_function

from six.moves import cPickle
import numpy as np
import scipy.misc
import itertools
import os
import yaml
cfg=yaml.safe_load(open('core/config.yaml'))
type_ex='color'
# configs for feature
n_bin   = cfg[type_ex]['n_bin']        # feature bins
n_slice = cfg[type_ex]['n_slice']         # slice image
h_type  = cfg[type_ex]['h_type']
class Color(object):
    ''' count img color feature
    
        arguments
            input    : a path to a image or a numpy.ndarray
            n_bin    : number of bins for each channel
            type     : 'global' means count the feature for whole image
                    'region' means count the feature for regions in images, then concatanate all of them
            n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
            normalize: normalize output feature
    
        return
            type == 'global'
            a numpy array with size n_bin ** channel
            type == 'region'
            a numpy array with size n_slice * n_slice * (n_bin ** channel)
        '''
    def __init__(self, n_bin=n_bin, type=h_type, n_slice=n_slice, normalize=True):
        self.n_bin=n_bin
        self.type=h_type
        self.n_slice=n_slice
        self.normalize=normalize
        self.dimension= self.n_bin**3*self.n_slice**2 if self.type=='region' else self.n_bin**3 
    def feature(self, input):
        if isinstance(input, np.ndarray):  # examinate input type
            img = input.copy()
        else:
            img = scipy.misc.imread(input, mode='RGB')
        height, width, channel = img.shape
        bins = np.linspace(0, 256, self.n_bin+1, endpoint=True)  # slice bins equally for each channel
    
        if self.type == 'global':
            hist = self._count_hist(img, self.n_bin, bins, channel)
    
        elif self.type == 'region':
            hist = np.zeros((self.n_slice, self.n_slice, self.n_bin ** channel))
            h_silce = np.around(np.linspace(0, height, self.n_slice+1, endpoint=True)).astype(int)
            w_slice = np.around(np.linspace(0, width, self.n_slice+1, endpoint=True)).astype(int)
    
            for hs in range(len(h_silce)-1):
                for ws in range(len(w_slice)-1):
                    img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
                    hist[hs][ws] = self._count_hist(img_r, self.n_bin, bins, channel)
    
        if self.normalize:
            hist /= np.sum(hist)
    
        return hist.flatten()
  
  
    def _count_hist(self, input, n_bin, bins, channel):
        img = input.copy()
        bins_idx = {key: idx for idx, key in enumerate(itertools.product(np.arange(n_bin), repeat=channel))}  # permutation of bins
        hist = np.zeros(n_bin ** channel)
    
        # cluster every pixels
        for idx in range(len(bins)-1):
            img[(input >= bins[idx]) & (input < bins[idx+1])] = idx
        # add pixels into bins
        height, width, _ = img.shape
        for h in range(height):
            for w in range(width):
                b_idx = bins_idx[tuple(img[h,w])]
                hist[b_idx] += 1
        return hist
  


if __name__ == "__main__":
    color = Color()
    input = '../../data/style/0_0_001.png'
    # test normalize
    # hist = color.feature(input, type='global')
    hist = color.feature(input)
    assert hist.sum() - 1 < 1e-9, "normalize false"
    print(color.dimension)
    input = '../../data/style/0_0_006.png'
    # test normalize
    # hist2 = color.feature(input, type='global')
    hist2 = color.feature(input)
    assert hist.sum() - 1 < 1e-9, "normalize false"
    # print(hist2)
    print(np.sum((hist - hist2) ** 2))