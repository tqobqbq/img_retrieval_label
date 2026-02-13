from PIL import Image
from feature_extractor import FeatureExtractor1,FeatureExtractor2
from pathlib import Path
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a',  type=int)
    args = parser.parse_args()
    configs=[['./static/feature1',FeatureExtractor1,5000],
             ['./static/feature2',FeatureExtractor2,5001]]
    config=configs[args.a]
    fe=config[1]()

    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path(config[0]) / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)
