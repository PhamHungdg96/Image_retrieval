import pandas as pd 
import numpy as np 
import ngtpy
import os
import shutil
import yaml

from color import Color
from HOG import HOG
from daisy import Daisy
from gabor import Gabor
from SIFT import SIFT
from resnet import Restnet_Ex
# genaral sample and query

cfg=yaml.safe_load(open('config.yaml'))

def general_sample_and_query():
    df=pd.read_csv('../../data/style/style.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    sample_l=int(0.8*len(df))

    sample_df, query_df=df[:sample_l],df[sample_l:]

    sample_df.to_csv(r'../index/samples.csv',index=None)
    query_df.to_csv(r'../index/querys.csv',index=None)

df_sample= pd.read_csv(r'../index/samples.csv')
def create_index(df_sample,type_ex='color'):
    assert type_ex in ['color', 'daisy','gabor','hog','sift','resnet']
    folder_img = '../../data/style/'
    path_index = '../index/%s'%type_ex
    if os.path.exists(path_index):
        shutil.rmtree(path_index)
        # return
    if type_ex=='color':
        ex=Color()
    elif type_ex=='daisy':
        ex=Daisy()
    elif type_ex=='gabor':
        ex=Daisy()
    elif type_ex=='hog':
        ex=HOG()
    elif type_ex=='sift':
        ex=SIFT()
    elif type_ex=='resnet':
        ex=Restnet_Ex()
    else: 
        print('not support type of extract')
        return
    ngtpy.create(path_index, dimension=ex.dimension, distance_type="Normalized Cosine")
    index_ngtpy = ngtpy.Index(path_index)
    for row in df_sample.values:
        hist=ex.feature(folder_img+row[-1])
        objectID = index_ngtpy.insert(hist)
        if objectID % 500 == 0:
            print('Processed {} objects.'.format(objectID))
    index_ngtpy.build_index()
    index_ngtpy.save()
    index_ngtpy.close()
    

if __name__=='__main__':
    create_index(df_sample, type_ex='resnet')