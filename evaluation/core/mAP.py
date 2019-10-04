import numpy as np
import ngtpy
import pandas as pd
import yaml
import plt

from color import Color
from HOG import HOG
from daisy import Daisy
from gabor import Gabor
from SIFT import SIFT
from resnet import Restnet_Ex

folder_img = '../../data/style/'
df_query= pd.read_csv(r'../index/querys.csv')
df_sample= pd.read_csv(r'../index/samples.csv')
cfg=yaml.safe_load(open('config.yaml'))
depth=10
def _query(input,type_ex='color',depth=10):
    assert type_ex in ['color', 'daisy','gabor','hog','sift','resnet']
    path_index = '../index/%s'%type_ex
    index_ngtpy = ngtpy.Index(path_index)
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
        
    ft=ex.feature(input)
    results=index_ngtpy.search(ft, size=depth)
    return [id for id,_ in results]

def mAP(db_query,db_sample,type_ex='color',depth=10):
    list_mAP=[]
    for id_lbl,(lbl,ip) in enumerate(db_query):
        id_result=_query(ip,type_ex=type_ex,depth=depth)
        lbl_result=np.array([db_sample[i][0] for i in id_result])
        m_ap=(lbl_result==lbl).sum()/depth
        list_mAP.append((m_ap,id_lbl,id_result))
    list_mAP=np.array(list_mAP)
    # print("best mAP: %s, bad mAP: %s, medium mAP: %s"%(np.max(list_mAP),np.min(list_mAP),np.mean(list_mAP)))
    return list_mAP



db_sample   =  [ (lbl,folder_img + nm_img) for lbl,nm_img in df_sample.iloc[:,-2:].values]
db_query    =  [ (lbl,folder_img + nm_img) for lbl,nm_img in df_query.iloc[:,-2:].values]
depth=10
type_ex='resnet'
list_mAP_exp=mAP(db_query,db_sample,type_ex=type_ex,depth=depth)
list_mAP=np.array(list_mAP_exp[:,0])
best_id=np.argmax(list_mAP)
bad_id=np.argmin(list_mAP)

summary="best mAP: %s, bad mAP: %s, medium mAP: %s"%(np.max(list_mAP),np.min(list_mAP),np.mean(list_mAP))
print(summary)
plt.show_summary(list_mAP,type_ex=type_ex,depth=depth,summary=summary,is_save=True)
print(list_mAP_exp[best_id,1:])
print(list_mAP_exp[bad_id,1:])
id_query,id_results=list_mAP_exp[best_id,1:]
print(db_query[id_query])
print([db_sample[id] for id in id_results])
plt.show_image_result(db_query[id_query], [db_sample[id] for id in id_results],type_ex=type_ex,depth_show=10, is_save=True, type_show='best')