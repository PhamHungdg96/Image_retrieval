from __future__ import print_function
from skimage.feature import hog
from skimage import color
import numpy as np
import os
import cv2
import yaml
cfg=yaml.safe_load(open('core/config.yaml'))
type_ex='sift'
# configs for feature
n_d   = cfg[type_ex]['n_d']
class SIFT(object):
    def __init__(self,n_d=n_d, normalize=True):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.n_d=n_d #number of descriptors
        self.normalize=normalize
        self.dimension=self.n_d * 64
    def feature(self, input):
        ''' SIFT extract feature in image
    
        arguments
            input    : a path to a image or a numpy.ndarray
            vector_size : size of extract feature vector
            normalize: normalize output histogram
    
        return
            vector
        '''
        if isinstance(input, np.ndarray):  # examinate input type
            img = input.copy()
        else:
            img = cv2.imread(input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps = self.sift.detect(img,None)
        kps = sorted(kps, key=lambda x: -x.response)[:self.n_d]
        kps, dsc = self.sift.compute(img, kps)
        dsc = dsc.flatten()
        if dsc.size < self.dimension:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(self.dimension - dsc.size)])
        dsc=dsc[:self.dimension]
        if self.normalize:
            dsc = dsc / np.sum(dsc, axis=0)
        return dsc

if __name__ == "__main__":
    _SIFT = SIFT()
    print(_SIFT.dimension)
    input = '../../data/style/0_0_001.png'
    hist = _SIFT.feature(input)
    input = '../../data/style/0_0_006.png'
    hist2 = _SIFT.feature(input)
    print(np.sum((hist - hist2) ** 2))
    