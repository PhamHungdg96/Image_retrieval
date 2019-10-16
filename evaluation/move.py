import shutil
import glob

imgs=glob.glob('data_folder/tablets/images/*.*')

for img in imgs:
    try:
        shutil.move(img,'data_folder/images/')
    except:
        print('trung lap: ',img)