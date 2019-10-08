import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.misc
def show_summary(y,type_ex=None,depth=10,summary=None,is_save=False,output_dir = "../../summary/"):
    x=range(len(y))
    f = plt.figure()
    title='Type of query: %s depth: %s'%(type_ex,depth)
    file_name='summary_%s_depth%s'%(type_ex,depth)
    f.suptitle(title,fontsize=16)
    f.canvas.manager.full_screen_toggle()
    f.canvas.set_window_title('%s'%file_name)
    ax = f.add_subplot(111)
    ax.bar(x, y)
    ax.set_xlabel('query')
    ax.set_ylabel('mAP')
    ax.set_title('%s'%summary)
    if not is_save:
        plt.show()  
    else: 
        output=os.path.join(output_dir,type_ex)
        if not os.path.exists(output):
            os.makedirs(output)
        f.savefig(os.path.join(output,'%s.png'%file_name))  
def show_image_result(query, results,type_ex=None, depth_show=10,is_save=False, type_show='best',output_dir = "../../summary/"):
    rows = 3
    cols = depth_show//(rows-1)
    depth=len(results)
    file_name='%s_result_%s_depth%s'%(type_show,type_ex,depth)
    f=plt.figure()
    f.canvas.manager.full_screen_toggle()
    for num  in range(rows*cols):
        if num==0:
            if isinstance(query[-1], np.ndarray):  # examinate input type
                img = query[-1].copy()
            else:
                img = scipy.misc.imread(query[-1], mode='RGB')
            ax = f.add_subplot(rows,cols,num+1)
            ax.set_title('query: %s'% query[1])
            ax.axis('off')
            ax.imshow(img,interpolation='nearest')
        elif num > cols-1:
            if num-cols<len(results):
                if isinstance(results[num-cols][1], np.ndarray):  # examinate input type
                    img = results[num-cols][1].copy()
                else:
                    img = scipy.misc.imread(results[num-cols][-1], mode='RGB')
                ax = f.add_subplot(rows,cols,num+1)
                ax.set_title(results[num-cols][1])
                ax.axis('off')
                ax.imshow(img, interpolation='nearest')
    if not is_save:
        plt.show()  
    else: 
        output=os.path.join(output_dir,type_ex)
        if not os.path.exists(output):
            os.makedirs(output)
        f.savefig(os.path.join(output,'%s.png'%file_name))
if __name__=='__main__':
    y=range(1000)
    show_summary(y, is_save=True)