from flask_restplus import Namespace, Resource, fields
from flask_api import status
from flask import request, send_from_directory
import base64
import numpy as np
import re
import io
from PIL import Image
import sys
sys.path.append('../..')
from init import net
import json
from aquiladb import AquilaClient as acl
import torch
import pandas as pd
db_remote = acl('0.0.0.0', 50051)

# import ngtpy
# index_ngtpy = ngtpy.Index(b"data/Indexdb")

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(150),
    transforms.RandomCrop(150),
    transforms.ToTensor()
])
# net=ResidualNet()
api = Namespace('image', description='post image to search and respone result')

image = api.model('Image', {
    'dataImage': fields.String(required=True, description='string of image'),
})
lb=['ref','brach','product']
df=pd.read_csv('data/style/style.csv')


@api.route('/')
class ImageClass(Resource):
    @api.expect(image)
    def post(self):
        json_data = request.json
        imgdata = re.sub('^data:image/.+;base64,', '', json_data['dataImage'])
        byte_data=base64.b64decode(imgdata)
        image = Image.open(io.BytesIO(byte_data)).convert('RGB')
        query=transform(image)
        vector_query=net(torch.unsqueeze(query, dim=0))['avg'].data[0].numpy()
        query_matrix = db_remote.convertMatrix(vector_query)
        results = db_remote.getNearest(query_matrix, 15)
        # results = index_ngtpy.search(vector_query, size=15)
        print('result:',results)
        res=[]
        # for idx, _ in results:
        for doc in json.loads(results.documents):
            output={k:v for k,v in zip(lb,doc['doc']['idx'].split(' || '))}
            output['ref']=str("/image/" + df.iloc[int(output['ref']),-1])
            # output={}
        #     output.update({'ref':str("/image/" + df.iloc[idx,-1])})
        #     output.update({'brach':str(df.iloc[idx,0])})
        #     output.update({'product':str(df.iloc[idx,2])})
            res.append(output)
        return json.dumps(res)
@api.route('/<name>')
@api.param('name', 'The name of image')
@api.response(404, 'Not found')
class ImagePath(Resource):
    @api.doc('get_image')
    def get(self, name):
        return send_from_directory('data/style', name)
        api.abort(404)