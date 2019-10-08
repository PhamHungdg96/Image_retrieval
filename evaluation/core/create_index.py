import pandas as pd 
import numpy as np 
import ngtpy
import os
import shutil
import yaml

from .color import Color
from .HOG import HOG
from .daisy import Daisy
from .gabor import Gabor
from .SIFT import SIFT
from .resnet import Restnet_Ex
# genaral sample and query

def general_sample_and_query():
    df=pd.read_csv('../../data/style/style.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    sample_l=int(0.8*len(df))

    sample_df, query_df=df[:sample_l],df[sample_l:]

    sample_df.to_csv(r'../index/samples.csv',index=None)
    query_df.to_csv(r'../index/querys.csv',index=None)

class CreateIndex(object):
    def __init__(self,db_sample,path_image,type_ex='color', path_index_root='../index'):
        self.type_ex=type_ex
        self.db_sample=db_sample
        self.path_image=path_image
        self.path_index=os.path.join(path_index_root,type_ex)
        self.ex=self.create_extractor(self.type_ex)
    @staticmethod
    def create_extractor(type_ex):
        assert type_ex in ['color', 'daisy','gabor','hog','sift','resnet'], 'not support type of extractor'
        if type_ex=='color':
            return Color()
        elif type_ex=='daisy':
            return Daisy()
        elif type_ex=='gabor':
            return Daisy()
        elif type_ex=='hog':
            return HOG()
        elif type_ex=='sift':
            return SIFT()
        elif type_ex=='resnet':
            return Restnet_Ex()
        else: 
            print('not support type of extractor')
            return None

    def __call__(self,**kwargs):
        ngtpy.create(self.path_index, dimension=self.ex.dimension,**kwargs)
        index_ngtpy = ngtpy.Index(self.path_index)
        for row in self.db_sample:
            hist=self.ex.feature(os.path.join(self.path_image,row[-1]))
            objectID = index_ngtpy.insert(hist)
            if objectID % 500 == 0:
                print('Processed {} objects.'.format(objectID))
        index_ngtpy.build_index()
        index_ngtpy.save()
        index_ngtpy.close()
    

if __name__=='__main__':
    df_sample= pd.read_csv(r'../index/samples.csv')
    create_index=CreateIndex(df_sample.values,path_image='../../data/style/', type_ex='color',path_index_root='../index')
    create_index(distance_type="Normalized Cosine")