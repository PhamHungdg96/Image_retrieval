from data.process import StyleDataset
from model.resnet import ResidualNet
import torch
import numpy as np 
from aquiladb import AquilaClient as acl
from torchvision import transforms
# import ngtpy
import os 
import shutil




transform = transforms.Compose([
    transforms.Resize(150),
    transforms.RandomCrop(150),
    transforms.ToTensor()
])
# db_remote = acl('0.0.0.0', 50051)
ds=StyleDataset('style.csv', root_dir='../data/style',transform=transform)

net=ResidualNet()
def index(net, ds):
    db_remote = acl('0.0.0.0', 50051)
    # if os.path.exists('data/Indexdb'):
    #     shutil.rmtree('data/Indexdb')
    # print('file in data: ',os.listdir('data'))
    print('start index......')
    # ngtpy.create(b"data/Indexdb", dimension=2048, distance_type="Normalized Cosine")
    # index_ngtpy = ngtpy.Index(b"data/Indexdb")
    docs_gen=[]
    batch_len=200
    for idx in range(len(ds)):
        x = ds[idx][0]
        y = ds[idx][1] + ' || ' + ds[idx][2]
        vector=net(torch.unsqueeze(x, dim=0))['avg'].data[0].numpy()
        docs_gen.append(db_remote.convertDocument(vector, {"idx":str(idx)+' || '+y}))
        # objectID = index_ngtpy.insert(vector)
        # if objectID % 500 == 0:
        #     print('Processed {} objects.'.format(objectID))
        if idx % batch_len == 0:
            # add documents to AquilaDB
            response=db_remote.addDocuments(docs_gen)
            print("index: "+str(idx), "inserted: "+str(len(response._id)))
            docs_gen = []
    # index_ngtpy.build_index()
    # index_ngtpy.save()
    # index_ngtpy.close()
if __name__=='__main__':
    index(net,ds)