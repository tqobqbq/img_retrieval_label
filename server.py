import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor1,FeatureExtractor2
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import argparse
app = Flask(__name__)

# Read image features


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print(request.form)
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        print(ids)
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a',  type=int)
    args = parser.parse_args()
    configs=[['./static/feature1',FeatureExtractor1,5000],
             ['./static/feature2',FeatureExtractor2,5001]]
    config=configs[args.a]
    fe=config[1]()
    features = []
    img_paths = []
    for feature_path in Path(config[0]).glob("*.npy"):
##        print(feature_path)
        features.append(np.load(feature_path))
        img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
    features = np.array(features)
    app.run(port=config[2])
