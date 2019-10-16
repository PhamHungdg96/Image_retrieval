import numpy as np
import ngtpy
import pandas as pd
import yaml
import time

from .plt import show_image_result,show_summary
import os
from .color import Color
from .HOG import HOG
from .daisy import Daisy
from .gabor import Gabor
from .SIFT import SIFT
from .resnet import Restnet_Ex

cfg=yaml.safe_load(open('core/config.yaml'))
class Evaluate(object):
    def __init__(self, type_ex='color', depth=10, path_index_root='../index'):
        self.type_ex=type_ex
        self.depth=depth
        self.index_ngtpy=ngtpy.Index(os.path.join(path_index_root,type_ex))
        self.ex=self.create_extractor(self.type_ex)
    
    def query(self,input,depth=None):
        ft= self.ex.feature(input)
        if depth is not None and depth != self.depth:
            return self.index_ngtpy.search(ft, depth)
        else: 
            return self.index_ngtpy.search(ft, self.depth)
    @staticmethod
    def create_extractor(type_ex):
        assert type_ex in ['color', 'daisy','gabor','hog','sift','resnet'], 'not support type of extractor'
        if type_ex=='color':
            return Color()
        elif type_ex=='daisy':
            return Daisy()
        elif type_ex=='gabor':
            return Gabor()
        elif type_ex=='hog':
            return HOG()
        elif type_ex=='sift':
            return SIFT()
        elif type_ex=='resnet':
            return Restnet_Ex()
        else: 
            print('not support type of extractor')
            return None

    def score_AP(self, db_query, db_sample):
        '''
        method return a list include: Average Precision (AP) each query, id of query, ids of result query
        parameter:
            db_query: list of query
            db_sample: data base of sample
        return: 
            a list of (AP,id query, ids of result query)
        '''
        list_AP=[]
        for id_lbl,(lbl,_,ip) in enumerate(db_query):
            results=self.query(ip)
            id_result=[id for id,_ in results]
            lbl_result=np.array([db_sample[i][0] for i in id_result]) #colum lbl
            _ap=(lbl_result==lbl).sum()/ self.depth
            list_AP.append((_ap,id_lbl,id_result))
            if id_lbl%500==0:
                print('Processed %s query'%id_lbl)
        list_AP=np.array(list_AP)
        # print("best mAP: %s, bad mAP: %s, medium mAP: %s"%(np.max(list_mAP),np.min(list_mAP),np.mean(list_mAP)))
        return list_AP
if __name__=='__main__':
    type_ex='color'
    depth=10
    folder_img = '../../data/style/'
    df_query= pd.read_csv(r'../index/querys.csv')
    df_sample= pd.read_csv(r'../index/samples.csv')
    eval=Evaluate(type_ex=type_ex,depth=depth)
    db_sample   =  [(lbl_id,lbl_nm,folder_img + nm_img) for lbl_nm,lbl_id,nm_img in df_sample.iloc[:,2:].values]
    db_query    =  [(lbl_id,lbl_nm,folder_img + nm_img) for lbl_nm,lbl_id,nm_img in df_query.iloc[:,2:].values]
    

    list_eval=eval.score_AP(db_query,db_sample)
    list_AP=np.array(list_eval[:,0])
    best_id=np.argmax(list_AP)
    bad_id=np.argmin(list_AP)

    summary="best AP: %s, bad AP: %s, medium AP(mAP): %s"%(np.max(list_AP),np.min(list_AP),np.mean(list_AP))
    print(summary)
    show_summary(list_AP,type_ex=type_ex,depth=depth,summary=summary,is_save=False)
    print(list_eval[best_id,1:])
    print(list_eval[bad_id,1:])
    id_best_query,id_best_results=list_eval[best_id,1:]
    print(db_query[id_best_query])
    print([db_sample[id] for id in id_best_results])
    show_image_result(db_query[id_best_query], [db_sample[id] for id in id_best_results],type_ex=type_ex,depth_show=10, is_save=False, type_show='best')