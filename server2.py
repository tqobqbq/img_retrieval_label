import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor1,FeatureExtractor2
from datetime import datetime
import time
from flask import Flask, request, render_template,jsonify
from pathlib import Path
import argparse
import shutil,os,json,requests,cv2
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
##from flask_cors import CORS
# Read image features

##@app.after_request
##def func_res(resp):     
##    res = make_response(resp)
##    res.headers['Access-Control-Allow-Origin'] = '*'
##    res.headers['Access-Control-Allow-Methods'] = 'GET,POST'
##    res.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
##    return res

def convert_image_path1(dst):
    dst_s=dst.split('.')
    assert len(dst_s)==2,dst
    a=0
    while True:
        new_dst=f'{dst_s[0]}_{a}.{dst_s[1]}'
        if not os.path.exists(new_dst):
            return  new_dst
        a+=1

        
def convert_image_path2(dst):
    dst_s=dst.split('.')
    assert len(dst_s)==2,dst
    a=0
    l=[]
    while True:
        new_dst=f'{dst_s[0]}_{a}.{dst_s[1]}'
        if not os.path.exists(new_dst):
            return  l
        l.append(new_dst)
        a+=1



def get_score(img):
    query = fe.extract(img)
####        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
##    print(query.shape,features.shape)
##    dists=-np.mean(features*query,axis=1)/np.sqrt((np.mean(features**2,axis=1)*np.mean(query**2)))
##    print(dists.shape)
##    ids = np.argsort(dists)[:30]  # Top 30 results
    sim=features@query
##    app.logger.info(str((sim.shape,sim.max(),sim.min(),sim.mean())))
    ids = np.argsort(-sim)[:30]
    
    scores = [(sim[id], '/'.join(['static','img2',l2[id][0]+'_'+l2[id][1]+'.jpg']),l2[id][2]) for id in ids]
    return scores

@app.route('/', methods=['GET', 'POST'])
def index():
    print('a')
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream).convert('RGB')  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        scores=get_score(img)

        return render_template('index1.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index1.html')

#
@app.route('/p', methods=[ 'GET','POST'])
def index2():
    if request.method == 'POST':
        data = request.get_json()
        shutil.move(data[0],convert_image_path1(data[1].replace('img2','result')))
        return jsonify([])
    else:
        filename=request.args.get('file',None)
        keywords=request.args.get('keywords',None)
        if filename:
            query = fe.extract(Image.open(filename))
            assert keywords,[filename,keywords]
            keywords=keywords.lower()
            scores=[]
            for j,i in enumerate(l2):
                if keywords in i[3].lower():
                    scores.append([features[j]@query,f'static/img2/{i[0]}_{i[1]}.jpg',i[2],i[3]])
            scores.sort(key=lambda x:-x[0])
            uploaded_img_path=filename
        else:
            img_req = requests.get('http://192.168.101.17:8080//' + 'shot.jpg')
            img_arr = np.array(bytearray(img_req.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, -1)
            img=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img =Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + 'a.jpg'
            img.save(uploaded_img_path)


            scores=get_score(img)

        return render_template('index1.html',
                               query_path=uploaded_img_path,
                               scores=scores)


@app.route('/a', methods=[ 'GET','POST'])
def index3():
    if request.method == 'POST':
        data = request.get_json()
        shutil.move(data[0],data[1].replace('img2','result'))
        return jsonify([])
    else:
        file_list=os.listdir('static/uploaded/')
        filename=request.args.get('file',None)
        keywords=request.args.get('keywords',None)

        if not filename:
            filename=file_list[0]
        index=file_list.index(filename)
        next_file=file_list[(index+1)%len(file_list)]
        scores=[]
        if keywords:
            keywords=keywords.lower()
            query = fe.extract(Image.open('static/uploaded/'+filename))
            for j,i in enumerate(l2):
                if keywords in i[3].lower():
                    scores.append([features[j]@query,f'static/img2/{i[0]}_{i[1]}.jpg',i[2],i[3]])
            scores.sort(key=lambda x:-x[0])
            
        else:
            scores=get_score(Image.open(os.path.join('static/uploaded/',filename)))
            
        
        return render_template('index3.html',
                               query_path='static/uploaded/'+filename,
                               scores=scores,
                               next_file=next_file)
if __name__=="__main__":
    data_dir=r'D:\marimite\doujin\doujinshidata'
    with open(os.path.join(data_dir,'data_merged.json'),'r',encoding='utf-8') as f:
        merged_data=json.load(f)
    print(len(merged_data))
    fe=FeatureExtractor2('checkpoint_0100.pth.tar')
    l1=[]
    l2=[]
    features = []
    for i,a in enumerate(merged_data):
        doujinshiorg_code=a['doujinshiorg_code']
        surugaya_code=a['surugaya_code']
        if surugaya_code:
            l1.append(['surugaya',surugaya_code,i,a['title']])
        if doujinshiorg_code:
            l1.append(['doujinshiorg',doujinshiorg_code,i,a['title']])

    feature_path=os.path.join('static','feature')
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)
    for i in l1:
        path=os.path.join(data_dir,i[0],'picture',i[1]+('_0' if i[0]=='doujinshiorg' else '')+'.jpg')
        if not os.path.exists(path):
            print(path)
            continue
        l2.append(i)
        path2=os.path.join('static','img2',i[0]+'_'+i[1]+'.jpg')
        if not os.path.exists(path2):
            shutil.copy(path,path2)
        path3=os.path.join(feature_path,i[0]+'_'+i[1]+'.npy')

        if os.path.exists(path3):
            feature=np.load(path3)
        else:
            feature=fe.extract(img=Image.open(path2))
            np.save(path3, feature)
        
##        feature=fe.extract(img=Image.open(path2))
        
        features.append(feature)
    print(len(l1),len(l2))
    features = np.array(features)
    print(features.shape)
    a=features@features.T
    print(a.shape,a.max(),a.min(),a.mean())
    app.run(port=5002,debug=True)
