import argparse
from core.create_index import CreateIndex
from core.mAP import Evaluate
from core.plt import show_summary,show_image_result,print_log
import pandas as pd
import numpy as np
import time

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--type", help="color, daisy,gabor,hog,sift,resnet",default='color', type=str)
    parser.add_argument("-d","--depth", help="size of results",default=10, type=int)
    parser.add_argument("-m","--mode", help="init,query,eval",default='init',type=str)
    parser.add_argument("-l","--log",help="file log",default='log',type=str)
    args = parser.parse_args()
    type_ex=args.type
    log_f=args.log
    log=print_log(output_dir=log_f,type_ex=type_ex)
    if args.mode == 'init':
        log(10*'-'+'mode create index'+10*'-')
        df_sample= pd.read_csv(r'index/samples.csv')
        create_index=CreateIndex(df_sample.values,path_image='data_folder/images/', type_ex=type_ex,path_index_root='index',log=log_f)
        create_index(distance_type="Normalized Cosine")
    elif args.mode == 'query':
        print('init')
    elif args.mode == 'eval':
        log(10*'-'+'mode evaluation query'+10*'-')
        depth=int(args.depth)
        if depth!=10: print('depth: %s'%depth)
        folder_img = 'data_folder/images/'
        output_dir='summary/'
        df_query= pd.read_csv(r'index/querys.csv')
        df_sample= pd.read_csv(r'index/samples.csv')

        eval=Evaluate(type_ex=type_ex,depth=depth,path_index_root='index')
        db_sample   =  [(lbl_id,lbl_nm,folder_img + nm_img) for lbl_nm,lbl_id,nm_img in df_sample.iloc[:,2:].values]
        db_query    =  [(lbl_id,lbl_nm,folder_img + nm_img) for lbl_nm,lbl_id,nm_img in df_query.iloc[:,2:].values]
        
        start_time=time.time()
        list_eval=eval.score_AP(db_query,db_sample)
        log('with size %d ,take about %.3f ms each query.'%(depth,(time.time()-start_time)*1000/len(db_query)))
        list_AP=np.array(list_eval[:,0])
        best_id=np.argmax(list_AP)
        bad_id=np.argmin(list_AP)
        best_percent=np.sum(list_AP==list_AP.max())/depth
        bad_percent=np.sum(list_AP==list_AP.min())/depth

        summary="best AP: %s (%s%%), bad AP: %s (%s%%), medium AP(mAP): %.3f"%(np.max(list_AP),best_percent,np.min(list_AP),bad_percent,np.mean(list_AP))
        print(summary); log(summary)
        show_summary(list_AP,type_ex=type_ex,depth=depth,summary=summary,is_save=True,output_dir=output_dir)
        print(list_eval[best_id,1:])
        print(list_eval[bad_id,1:])
        
        id_best_query,id_best_results=list_eval[best_id,1:]
        print(db_query[id_best_query])
        print([db_sample[id] for id in id_best_results])
        show_image_result(db_query[id_best_query], [db_sample[id] for id in id_best_results],type_ex=type_ex,depth_show=10, is_save=True, type_show='best',output_dir=output_dir)

        id_bad_query,id_bad_results=list_eval[bad_id,1:]
        print(db_query[id_bad_query])
        print([db_sample[id] for id in id_bad_results])
        show_image_result(db_query[id_bad_query], [db_sample[id] for id in id_bad_results],type_ex=type_ex,depth_show=10, is_save=True, type_show='bad',output_dir=output_dir)
    else:
        print('not support')

