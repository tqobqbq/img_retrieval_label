import numpy as np
from PIL import Image
import torch
from feature_extractor import FeatureExtractor1,FeatureExtractor2,FeatureExtractor5, FeatureExtractor6
from datetime import datetime
import time
from flask import Flask, request, render_template,jsonify, send_file
from pathlib import Path
import argparse
import shutil,os,json,requests,cv2
import logging
import io
import base64
import threading
from collections import OrderedDict
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

def move_img(src_img,dst_img):
    _,postfix=src_img.split('.')
    if 'surugaya' in dst_img:
        site='surugaya'
        assert 'doujinshiorg' not in dst_img,(dst_img,src_img)
    elif 'doujinshiorg' in dst_img:
        site='doujinshiorg'
    else:
        raise ValueError((dst_img,src_img))
    
    code=os.path.basename(dst_img).split('_')[0].split('.')[0]
    a=0
    while True:
        new_dst=f'static/result/{site}_{code}_{a}.{postfix}'
        if not os.path.exists(new_dst):
            return  shutil.move(src_img,new_dst)
        a+=1
# def convert_image_path1(dst):
#     dst_s=dst.split('.')
#     assert len(dst_s)==2,dst
#     a=0
#     while True:
#         new_dst=f'{dst_s[0]}_{a}.{dst_s[1]}'
#         if not os.path.exists(new_dst):
#             return  new_dst
#         a+=1

        
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

def get_score(img,keyword=None):
    return new_model.search(img,keyword=keyword)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # file = request.files['query_img']
        filepath=request.form.get('filepath',None)
        filename=filepath.split('/')[-1].split('\\')[-1]
        # img=Image.open(filepath).convert('RGB')
        # # Save query image
        # img = Image.open(file.stream).convert('RGB')  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + filename
        # img.save(uploaded_img_path)
        shutil.copy2(filepath,uploaded_img_path)
        scores=get_score(filepath)

        return render_template('index1.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index_unified.html')

@app.route('/img', methods=['GET'])
def get_image():
    filename=request.args.get('file',None)
    if filename and os.path.exists(filename):
        return send_file(filename, mimetype='image/jpeg')
    else:
        return jsonify({'error': 'File not found'}), 404
#
@app.route('/p', methods=[ 'GET','POST'])
def index2():
    if request.method == 'POST':
        data = request.get_json()
        # shutil.move(data[0],convert_image_path1(data[1].replace('img2','result')))
        move_img(data[0],data[1])
        return jsonify([])
    else:
        filename=request.args.get('file',None)
        keywords=request.args.get('keywords',None)
        assert keywords,[filename,keywords]
        # if filename:
        #     dis=[i.item() for i in new_model.cal(Image.open(filename))]
        # else:
        #     dis=[0 for _ in range(len(new_model.filenames))]
        keywords=keywords.lower()
        uploaded_img_path=filename
        scores=get_score(filename,keyword=keywords)
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
        # shutil.move(data[0],data[1].replace('img2','result'))
        move_img(data[0],data[1])
        return jsonify([])
    else:
        file_list=os.listdir('static/uploaded/')
        filename=request.args.get('file',None)
        keywords=request.args.get('keywords',None)

        if not filename:
            filename=file_list[0]
        index=file_list.index(filename)
        next_file=file_list[(index+1)%len(file_list)]
        filepath=os.path.join('static','uploaded',filename)
        scores=get_score(filepath,keyword=keywords)
            
        next_path = os.path.join('static','uploaded',next_file)
        new_model.precompute_dis_async(next_path)
        return render_template('index3.html',
                               query_path='static/uploaded/'+filename,
                               scores=scores,
                               next_file=next_file)
    

@app.route('/api/search', methods=['POST'])
def api_search():
    filepath = request.form.get('filepath', None)
    keyword = request.form.get('keyword', None)
    query_path_existing = request.form.get('query_path', None)
    file = request.files.get('query_img', None)

    query_path = None
    next_file = None

    if filepath:
        # Method 1: File path upload - copy to uploaded and find next file
        filename = filepath.split('/')[-1].split('\\')[-1]
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + filename
        shutil.copy2(filepath, uploaded_img_path)
        query_path = filepath

        # Find next image in same directory
        try:
            parent_dir = os.path.dirname(filepath)
            siblings = sorted([f for f in os.listdir(parent_dir)
                               if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.gif','.webp'))])
            basename = os.path.basename(filepath)
            if basename in siblings:
                idx = siblings.index(basename)
                next_name = siblings[(idx + 1) % len(siblings)]
                next_file = os.path.join(parent_dir, next_name)
                new_model.precompute_dis_async(next_file)
        except Exception:
            pass
    elif file:
        # Method 2: Whole image file upload
        img = Image.open(file.stream).convert('RGB')
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        query_path = uploaded_img_path
    elif query_path_existing:
        # Keyword re-search with existing query path
        query_path = query_path_existing

    if keyword and not query_path:
        # Keyword-only search
        scores = new_model.search(None, keyword=keyword)
    elif keyword:
        scores = new_model.search(query_path, keyword=keyword)
    else:
        scores = get_score(query_path)

    results = [{'score': s[0], 'path': s[1], 'title': s[2] if len(s) > 2 else '', 'info': s[3] if len(s) > 3 else ''} for s in scores]
    resp = {'query_path': query_path or '', 'results': results}
    if next_file:
        resp['next_file'] = next_file
    return jsonify(resp)


@app.route('/api/upload_image', methods=['POST'])
def api_upload_image():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
    img.save(uploaded_img_path)
    return jsonify({'path': uploaded_img_path})


@app.route('/api/search_region', methods=['POST'])
def api_search_region():
    file = request.files['region_img']
    region_info = request.form.get('region_info', '{}')
    img = Image.open(file.stream).convert('RGB')

    # Save cropped region
    uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_region.jpg"
    img.save(uploaded_img_path)

    scores = get_score(img)
    results = [{'score': s[0], 'path': s[1], 'title': s[2] if len(s) > 2 else '', 'info': s[3] if len(s) > 3 else ''} for s in scores]

    # Generate base64 preview of the cropped region
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    cropped_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        'query_path': uploaded_img_path,
        'cropped_image': 'data:image/jpeg;base64,' + cropped_b64,
        'results': results
    })

class model:
    def __init__(self,device='cpu'):
        self.name_extractor2={'dinov2_vitl14_pretrain':FeatureExtractor5(device=device),'dinov3_vith16plus':FeatureExtractor6(device=device),'resnet18_simclr_checkpoint_0100':FeatureExtractor2('checkpoint_0100.pth.tar',device=device)}

        data=torch.load('features.pt')

        self.filenames=data['filenames']
        self.titles=[]

        self.name_map={}

        for i,d in enumerate(merged_data):
            for j in ['surugaya','doujinshiorg']:
                k=d[f'{j}_code']
                if k:
                    self.name_map[(j,k.lower())]=i
        for filename in self.filenames:
            a=filename.split('/')
            b=a[-3]
            assert b in ['surugaya','doujinshiorg'],f'{filename} {a}'
            c=a[-1].split('_')[0].split('.')[0]
            
            self.titles.append(merged_data[self.name_map[(b,c.lower())]]['title'])
        self.func=lambda x:(x-x.mean(dim=1,keepdim=True))/(x.std(dim=1,keepdim=True)+1e-8)
        self.func3=lambda a,b:-torch.cosine_similarity(a, b, dim=-1)
        self.name_features={k:self.func(v[:len(self.filenames)])[:,None] for k,v in data['name_features'].items() if k in self.name_extractor2}

        self.data_dir=r'D:\marimite\doujin\doujinshidata'


        
        self.dis_cache = OrderedDict()
        self.caling_set=set()
        self.cache_size = 5
        self.cache_lock = threading.Lock()

    def cal(self,img):
        if isinstance(img, str):
            with self.cache_lock:
                if img in self.dis_cache:
                    return self.dis_cache[img]
            if img in self.caling_set:
                count=0
                while True:
                    with self.cache_lock:
                        if img in self.dis_cache:
                            return self.dis_cache[img]
                    time.sleep(0.1)
                    count+=1
                    if count>300:
                        raise TimeoutError(f'cal timeout for {img}')
                    
            with self.cache_lock:
                self.caling_set.add(img)
            dis=0
            img2=Image.open(img)
            for name,fe in self.name_extractor2.items():
                dis+=self.func3(self.name_features[name], self.func(fe.extract(img2,rotate=True))[None])
            res=dis.min(dim=1,keepdim=True).values
            with self.cache_lock:
                self.caling_set.discard(img)
                self.dis_cache[img]=res
                if len(self.dis_cache)>self.cache_size:
                    self.dis_cache.popitem(last=False)
        else:
            dis=0
            img2=img
            for name,fe in self.name_extractor2.items():
                dis+=self.func3(self.name_features[name], self.func(fe.extract(img2,rotate=True))[None])
            res=dis.min(dim=1,keepdim=True).values
        return res
    
    def precompute_dis_async(self,img):
        threading.Thread(target=self.cal,args=(img,)).start()
    def search(self,img,num=30,keyword=None):
        assert keyword or img, 'keyword or img should be provided'
        if img:
            dis=self.cal(img)
        else:
            dis=[0 for _ in range(len(self.filenames))]
        if keyword:
            keyword=keyword.lower()
            ids=[i for i,title in enumerate(self.titles) if keyword in title.lower()]
        else:
            ids = torch.argsort(dis,dim=0,descending=False)[:num,0]
        if img:
            dis=[i.item() for i in dis]
        # print(ids.shape,dis.shape,ids)
        print(len(ids))
        scores=[]
        for id_ in ids:
            filename=self.filenames[id_]
            a=filename.split('/')
            b=a[-3]
            assert b in ['surugaya','doujinshiorg'],f'{filename} {a}'
            c=a[-1].split('_')[0].split('.')[0]
            i=self.name_map[(b,c.lower())]
            d=merged_data[i]
            title=d['title']
            s=''
            if d['surugaya_code']:
                s=f'{surugaya_data[d["surugaya_code"]]["date_price"].get(surugaya_sorted_date[-1], None)} {bought_items.get(d["surugaya_code"],None)} '
            s=f'{s} {d["surugaya_code"] and d["surugaya_code"] in bought_items2} {bought_filenames.get(d["surugaya_code"],[])} {bought_filenames.get(d["doujinshiorg_code"],[])} {d['surugaya_code']} {d["doujinshiorg_code"]} {i}'
            scores.append([dis[id_], os.path.join(self.data_dir,filename),title,s])
            # scores.append([dis[id_], '/'.join(['static','img2',b+'_'+c+'.jpg']),title,s])
        scores.sort(key=lambda x:x[0],reverse=False)#from small to large
        return scores
        
if __name__=="__main__":
    data_dir=r'D:\marimite\doujin\doujinshidata'
    with open(os.path.join(data_dir,'data_merged.json'),'r',encoding='utf-8') as f:
        merged_data=json.load(f)
    print(len(merged_data))
    new_model=model('cpu')
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
    app.run(port=5002,debug=True)
