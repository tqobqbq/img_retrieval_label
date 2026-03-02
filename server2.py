import numpy as np
from PIL import Image
import torch
from feature_extractor import FeatureExtractor1,FeatureExtractor2,FeatureExtractor5, FeatureExtractor6
from datetime import datetime
import time
from flask import Flask, request, render_template,jsonify
from pathlib import Path
import argparse
import shutil,os,json,requests,cv2
import logging
import io
import base64

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



# def get_score(img):
#     query = fe.extract(img)
# ####        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
# ##    print(query.shape,features.shape)
# ##    dists=-np.mean(features*query,axis=1)/np.sqrt((np.mean(features**2,axis=1)*np.mean(query**2)))
# ##    print(dists.shape)
# ##    ids = np.argsort(dists)[:30]  # Top 30 results
#     sim=features@query
# ##    app.logger.info(str((sim.shape,sim.max(),sim.min(),sim.mean())))
#     ids = np.argsort(-sim)[:30]
    
#     scores = [(sim[id], '/'.join(['static','img2',l2[id][0]+'_'+l2[id][1]+'.jpg']),l2[id][2]) for id in ids]
#     return scores
def get_score(img):
    return new_model.search(img)

@app.route('/', methods=['GET', 'POST'])
def index():
    print('a')
    if request.method == 'POST':
        # file = request.files['query_img']
        filepath=request.form.get('filepath',None)
        filename=filepath.split('/')[-1].split('\\')[-1]
        img=Image.open(filepath).convert('RGB')
        # # Save query image
        # img = Image.open(file.stream).convert('RGB')  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + filename
        # img.save(uploaded_img_path)
        shutil.copy2(filepath,uploaded_img_path)
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
        assert keywords,[filename,keywords]
        if filename:
            dis=[i.item() for i in new_model.cal(Image.open(filename))]
        else:
            dis=[0 for _ in range(len(new_model.filenames))]
        keywords=keywords.lower()
        scores=[]
        uploaded_img_path=filename
        for j,d in enumerate(merged_data):
            title=d['title']
            if keywords not in title.lower():
                continue
            print(title,[d['surugaya_code'],d['doujinshiorg_code']])
            s=''
            if d['surugaya_code'] and d['surugaya_code'] in bought_items:
                s=f'{bought_items[d["surugaya_code"]]} '
            s=f'{s} {d["surugaya_code"] and d["surugaya_code"] in bought_items2} {d["surugaya_code"] and d["surugaya_code"] in bought_items2} {bought_filenames.get(d["surugaya_code"],[])} {bought_filenames.get(d["doujinshiorg_code"],[])}  {d['surugaya_code']} {d["doujinshiorg_code"]} {j}'
            for i in [d['surugaya_code'],d['doujinshiorg_code']]:
                if i:
                    if i in surugaya_data:
                        s=f'{surugaya_data[i]["date_price"].get(surugaya_sorted_date[-1], None)} {s} '
                    for id_,filename in enumerate(new_model.filenames):
                        a=filename.split('/')
                        b=a[-3]
                        assert b in ['surugaya','doujinshiorg'],f'{filename} {a}'
                        c=a[-1].split('_')[0].split('.')[0]
                        if c.lower()==i.lower():
                            scores.append([dis[id_].item(), '/'.join(['static','img2',b+'_'+c+'.jpg']),title,s])
                            break
                    else:
                        print((i,keywords,d))
                        scores.append([dis[id_].item(), None,title,s])
        print(len(scores))
        scores.sort(key=lambda x:-x[0])
        # else:
        #     img_req = requests.get('http://192.168.101.17:8080//' + 'shot.jpg')
        #     img_arr = np.array(bytearray(img_req.content), dtype=np.uint8)
        #     img = cv2.imdecode(img_arr, -1)
        #     img=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #     img =Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        #     uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + 'a.jpg'
        #     img.save(uploaded_img_path)


        #     scores=get_score(img)

        return render_template('index1.html',
                               query_path=uploaded_img_path,
                               scores=scores)


@app.route('/region', methods=['GET'])
def region_index():
    return render_template('index_region.html')

@app.route('/region/upload', methods=['POST'])
def region_upload():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
    img.save(uploaded_img_path)
    return jsonify({'path': uploaded_img_path})

@app.route('/region/search', methods=['POST'])
def region_search():
    data = request.get_json()
    image_path = data['image_path']
    regions = data['regions']
    
    img = Image.open(image_path).convert('RGB')
    results = []
    
    for i, region in enumerate(regions):
        x0, y0, x1, y1 = region['x0'], region['y0'], region['x1'], region['y1']
        # 裁剪区域
        cropped_img = img.crop((x0, y0, x1, y1))
        
        # 将裁剪后的图片转为base64
        buffered = io.BytesIO()
        cropped_img.save(buffered, format="JPEG")
        cropped_img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # 获取该区域的检索分数
        scores = get_score(cropped_img)
        
        results.append({
            'region': region,
            'cropped_image': f'data:image/jpeg;base64,{cropped_img_base64}',
            'scores': [[float(s[0]), *s[1:]] for s in scores[:20]]  # 取前20个结果
        })
    
    return jsonify({'results': results})

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
                    scores.append([features[j]@query,f'static/img2/{i[0]}_{i[1]}.jpg',i[2],i[3],s])
            scores.sort(key=lambda x:-x[0])
            
        else:
            scores=get_score(Image.open(os.path.join('static/uploaded/',filename)))
            
        
        return render_template('index3.html',
                               query_path='static/uploaded/'+filename,
                               scores=scores,
                               next_file=next_file)
    
class model:
    def __init__(self,device='cpu'):
        self.name_extractor2={'dinov2_vitl14_pretrain':FeatureExtractor5(device=device),'dinov3_vith16plus':FeatureExtractor6(device=device),'resnet18_simclr_checkpoint_0100':FeatureExtractor2('checkpoint_0100.pth.tar',device=device)}

        data=torch.load('features.pt')

        self.filenames=data['filenames']
        self.func=lambda x:(x-x.mean(dim=1,keepdim=True))/(x.std(dim=1,keepdim=True)+1e-8)
        self.func3=lambda a,b:-torch.cosine_similarity(a, b, dim=-1)
        self.name_features={k:self.func(v[:len(self.filenames)])[:,None] for k,v in data['name_features'].items() if k in self.name_extractor2}

        self.data_dir=r'D:\marimite\doujin\doujinshidata'


    def cal(self,img):
        dis=0
        for name,fe in self.name_extractor2.items():
            dis+=self.func3(self.name_features[name], self.func(fe.extract(img)[None])[None])
        # print(self.name_features[name][:5,:5],self.func(fe.extract(img)[None])[:5,:5])
        return dis
    
    def search(self,img,num=30):
        dis=self.cal(img)
        ids = torch.argsort(dis,dim=0,descending=False)[:num,0]
        print(ids.shape,dis.shape,ids)
        scores=[]
        for id in ids:
            filename=self.filenames[id]
            a=filename.split('/')
            b=a[-3]
            assert b in ['surugaya','doujinshiorg'],f'{filename} {a}'
            c=a[-1].split('_')[0].split('.')[0]
            for i,d in enumerate(merged_data):
                if (d['surugaya_code'] and d['surugaya_code'].lower()==c) or (d['doujinshiorg_code'] and d['doujinshiorg_code'].lower()==c):
                    title=d['title']
                    s=''
                    if d['surugaya_code']:
                        s=f'{surugaya_data[d["surugaya_code"]]["date_price"].get(surugaya_sorted_date[-1], None)} {bought_items.get(d["surugaya_code"],None)} '
                    s=f'{s} {d["surugaya_code"] and d["surugaya_code"] in bought_items2} {bought_filenames.get(d["surugaya_code"],[])} {bought_filenames.get(d["doujinshiorg_code"],[])} {d['surugaya_code']} {d["doujinshiorg_code"]} {i}'
                    scores.append([dis[id].item(), '/'.join(['static','img2',b+'_'+c+'.jpg']),title,s])
                    break
            else:
                raise ValueError(f'{c} {filename} not found in merged_data')
        return scores
        
new_model=model('cpu')
if __name__=="__main__":
    data_dir=r'D:\marimite\doujin\doujinshidata'
    with open(os.path.join(data_dir,'data_merged.json'),'r',encoding='utf-8') as f:
        merged_data=json.load(f)
    print(len(merged_data))
    with open(r'D:\chrome_download\surugaya_all_bought.json','r',encoding='utf-8') as f:
        bought_data=json.load(f)
    bought_items=bought_data['bought_items']
    bought_items2=set(bought_data['bought_items2'])
    bought_items3=bought_items2.union(bought_items)
    
    bought_filenames={}
    for i in os.listdir('static/result/'):
        a=i.split('_')[1]
        if a not in bought_filenames:
            bought_filenames[a]=[]
        bought_filenames[a].append(i)

    with open(r'd:\marimite\doujin\doujinshidata\surugaya\data.json','r',encoding='utf-8') as f:
        surugaya_data=json.load(f)
    surugaya_date_set=set()
    for i in surugaya_data.values():
        surugaya_date_set.update(i['date_price'])

    surugaya_sorted_date=sorted(surugaya_date_set)
#     fe=FeatureExtractor2('checkpoint_0100.pth.tar','cpu')
#     l1=[]
#     l2=[]
#     features = []
#     for i,a in enumerate(merged_data):
#         doujinshiorg_code=a['doujinshiorg_code']
#         surugaya_code=a['surugaya_code']
#         if surugaya_code:
#             l1.append(['surugaya',surugaya_code,i,a['title']])
#         if doujinshiorg_code:
#             l1.append(['doujinshiorg',doujinshiorg_code,i,a['title']])

#     feature_path=os.path.join('static','feature')
#     if not os.path.exists(feature_path):
#         os.mkdir(feature_path)
#     for i in l1:
#         path=os.path.join(data_dir,i[0],'picture',i[1]+('_0' if i[0]=='doujinshiorg' else '')+'.jpg')
#         if not os.path.exists(path):
#             print(path)
#             continue
#         l2.append(i)
#         path2=os.path.join('static','img2',i[0]+'_'+i[1]+'.jpg')
#         if not os.path.exists(path2):
#             shutil.copy(path,path2)
#         path3=os.path.join(feature_path,i[0]+'_'+i[1]+'.npy')

#         if os.path.exists(path3):
#             feature=np.load(path3)
#         else:
#             feature=fe.extract(img=Image.open(path2))
#             np.save(path3, feature)
        
# ##        feature=fe.extract(img=Image.open(path2))
        
#         features.append(feature)
#     print(len(l1),len(l2))
#     features = np.array(features)
#     print(features.shape)
#     a=features@features.T
#     print(a.shape,a.max(),a.min(),a.mean())
    app.run(port=5002,debug=True)
