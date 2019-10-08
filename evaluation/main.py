import argparse
from core.create_index import CreateIndex
from core.mAP import Evaluate
from core.plt import show_summary,show_image_result
import pandas as pd
import numpy as np
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--type", help="color, daisy,gabor,hog,sift,resnet",default='color', type=str)
    parser.add_argument("-d","--depth", help="size of results",default=10, type=int)
    parser.add_argument("-m","--mode", help="init,query,eval",default='init',type=str)
    args = parser.parse_args()
    type_ex=args.type
    if args.mode == 'init':
        df_sample= pd.read_csv(r'index/samples.csv')
        create_index=CreateIndex(df_sample.values,path_image='../data/style/', type_ex=type_ex,path_index_root='index')
        create_index(distance_type="Normalized Cosine")
    elif args.mode == 'query':
        print('init')
    elif args.mode == 'eval':
        depth=int(args.depth)
        if depth!=10: print('depth: %s'%depth)
        folder_img = '../data/style/'
        output_dir='summary/'
        df_query= pd.read_csv(r'index/querys.csv')
        df_sample= pd.read_csv(r'index/samples.csv')

        eval=Evaluate(type_ex=type_ex,depth=depth,path_index_root='index')
        db_sample   =  [(lbl_id,lbl_nm,folder_img + nm_img) for lbl_nm,lbl_id,nm_img in df_sample.iloc[:,2:].values]
        db_query    =  [(lbl_id,lbl_nm,folder_img + nm_img) for lbl_nm,lbl_id,nm_img in df_query.iloc[:,2:].values]
        

        list_eval=eval.score_AP(db_query,db_sample)
        list_AP=np.array(list_eval[:,0])
        best_id=np.argmax(list_AP)
        bad_id=np.argmin(list_AP)

        summary="best AP: %s, bad AP: %s, medium AP(mAP): %s"%(np.max(list_AP),np.min(list_AP),np.mean(list_AP))
        print(summary)
        show_summary(list_AP,type_ex=type_ex,depth=depth,summary=summary,is_save=True,output_dir=output_dir)
        print(list_eval[best_id,1:])
        print(list_eval[bad_id,1:])
        id_best_query,id_best_results=list_eval[best_id,1:]
        print(db_query[id_best_query])
        print([db_sample[id] for id in id_best_results])
        show_image_result(db_query[id_best_query], [db_sample[id] for id in id_best_results],type_ex=type_ex,depth_show=10, is_save=True, type_show='best',output_dir=output_dir)
    else:
        print('not support')

