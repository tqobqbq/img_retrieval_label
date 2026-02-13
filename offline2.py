from PIL import Image
from feature_extractor import FeatureExtractor1,FeatureExtractor2
from pathlib import Path
import numpy as np
import argparse
import json

if __name__ == '__main__':
    data_dir='D:\marimite\doujin\doujinshidata'
    with open(os.path.join(data_dir,'merge_data.json'),'r',encoding='utf-8') as f:
        merged_data=json.load(f)
    print(len(merged_data))
    img_list=[]
    for a in merged_data:
        b=[]
        doujinshiorg_code=a['doujinshiorg_code']
        surugaya_code=a['surugaya_code']
        if surugaya_code:
            path=os.path.join(data_dir,'surugaya','picture',surugaya_code+'.jpg')
            if os.path.exists(path):
                b.append(path)
        if doujinshiorg_code:
            path=os.path.join(data_dir,'doujinshiorg','picture',doujinshiorg_code+'_0.jpg')
            if os.path.exists(path):
                b.append(path)
        if len(b)>0:
            img_list.append(b)
    return img_list
