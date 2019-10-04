from __future__ import print_function
from skimage.feature import hog
from skimage import color
import numpy as np
import scipy.misc
import os
import yaml
cfg=yaml.safe_load(open('config.yaml'))
type_ex='hog'
# configs for feature
n_bin    = cfg[type_ex]['n_bin']
n_slice  = cfg[type_ex]['n_slice']
n_orient = cfg[type_ex]['n_orient']
p_p_c    = tuple(int(i) for i in cfg[type_ex]['p_p_c'].split(' '))
c_p_b    = tuple(int(i) for i in cfg[type_ex]['c_p_b'].split(' '))
h_type   = cfg[type_ex]['h_type']
class HOG(object):
    ''' count img histogram
    
        arguments
            input    : a path to a image or a numpy.ndarray
            n_bin    : number of bins of histogram
            type     : 'global' means count the feature for whole image
                    'region' means count the feature for regions in images, then concatanate all of them
            n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
            normalize: normalize output feature
    
        return
            type == 'global'
            a numpy array with size n_bin
            type == 'region'
            a numpy array with size n_bin * n_slice * n_slice
    '''
    def __init__(self,n_bin=n_bin, type=h_type, n_slice=n_slice, normalize=True):
        self.n_bin  = n_bin
        self.type   = h_type
        self.n_slice    = n_slice
        self.normalize  = normalize
        self.dimension  = self.n_bin if self.type=='global' else self.n_bin*self.n_slice*n_slice
    def feature(self, input):
        if isinstance(input, np.ndarray):  # examinate input type
            img = input.copy()
        else:
            img = scipy.misc.imread(input, mode='RGB')
        height, width, channel = img.shape
  
        if self.type == 'global':
            hist = self._HOG(img, self.n_bin)
  
        elif self.type == 'region':
            hist = np.zeros((self.n_slice, self.n_slice, self.n_bin))
            h_silce = np.around(np.linspace(0, height, self.n_slice+1, endpoint=True)).astype(int)
            w_slice = np.around(np.linspace(0, width, self.n_slice+1, endpoint=True)).astype(int)
  
            for hs in range(len(h_silce)-1):
                for ws in range(len(w_slice)-1):
                    img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
                    hist[hs][ws] = self._HOG(img_r, self.n_bin)
  
        if self.normalize:
            hist /= np.sum(hist)
  
        return hist.flatten()

    def _HOG(self, img, n_bin, normalize=True):
        image = color.rgb2gray(img)
        fd = hog(image, orientations=n_orient, pixels_per_cell=p_p_c, cells_per_block=c_p_b)
        bins = np.linspace(0, np.max(fd), n_bin+1, endpoint=True)
        hist, _ = np.histogram(fd, bins=bins)
  
        if normalize:
            hist = np.array(hist) / np.sum(hist)
  
        return hist


if __name__ == "__main__":
    _hog = HOG()
    input = '../../data/style/0_0_001.png'
    # test normalize
    # hist = color.histogram(input, type='global')
    hist = _hog.feature(input)
    assert hist.sum() - 1 < 1e-9, "normalize false"
    print(_hog.dimension)
    print(hist.__len__())
    input = '../../data/style/0_0_006.png'
    # test normalize
    # hist2 = color.histogram(input, type='global')
    hist2 = _hog.feature(input)
    assert hist.sum() - 1 < 1e-9, "normalize false"
    # print(hist2)
    print(np.sum((hist - hist2) ** 2))
