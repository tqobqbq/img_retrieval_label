
import numpy as np
import sys
sys.path.append(r'D:\pycode\SimCLR-master\models')
from resnet_simclr import ResNetSimCLR
from torchvision.transforms import transforms
import torchvision.transforms.v2 as transforms
import torch
from gaussian_blur import GaussianBlur
# See https://keras.io/api/applications/ for details

class FeatureExtractor1:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)

        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize



class FeatureExtractor2:
    def __init__(self,path):
##        base_model = VGG16(weights='imagenet')
##        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        s=1
        color_jitter =transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        size=96
##        self.transforms=transforms.Compose([transforms.Resize(size=size),
####                                              transforms.RandomResizedCrop(size=size),
####                                              transforms.RandomHorizontalFlip(),
####                                              transforms.RandomApply([color_jitter], p=0.8),
####                                              transforms.RandomGrayscale(p=0.2),
##                                              GaussianBlur(kernel_size=int(0.1 * size)),
##                                              transforms.ToTensor()])
##        self.transforms = transforms.Compose([
####                                            transforms.RandomRotation(180),
####                                            transforms.RandomResizedCrop(scale=(0.8,1.5),ratio=(0.8,1.25),size=size),
##                                            
####                                              transforms.RandomHorizontalFlip(),
####                                              transforms.RandomApply([color_jitter], p=0.8),
####                                              transforms.RandomGrayscale(p=0.2),
####                                              GaussianBlur(kernel_size=int(0.1 * size)),
##                                              transforms.Resize(size),
##                                              transforms.ToTensor()])
        self.transforms=transforms.Compose([
                                    transforms.Resize(size=(size,size)),
                                    # transforms.RandomResizedCrop(scale=(0.8,1.5),ratio=(0.8,1.25),size=size),
                                    transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
                                    ])
        self.model=ResNetSimCLR('resnet18',128)
        self.model.load_state_dict(torch.load(path)['state_dict'])
        self.model.eval()

    def extract(self, img):
##        print(type(img))
        input=self.transforms(img.convert('RGB'))
##        if input.shape==1:
##
        with torch.no_grad():
            output=torch.nn.functional.normalize(self.model(input.unsqueeze(0)),dim=1)[0]
        return output.detach().numpy()


##class FeatureExtractor3:
##    def __init__(self,device):
##        self.device=device
##        size=224
##        self.model="resnet18": models.resnet18(pretrained=use_pretrain, num_classes=1000)
##        self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, 128))
##        self.model=self.model.to(self.device)
##        model_path='checkpoint_0.pth.tar'
##        if os.path.exists(model_path):
##            model=torch.load(model_path)
##            self.model.load_state_dict(model['state_dict'])
##            self.a=model['a'].to(self.device)
##            self.b=model['b'].to(self.device)
##            self.feature_t=model['feature'].to(self.device)
##            self.l2=model['img_list']
##        else:
##            self.a=torch.zeros((1,128)).to(self.device)
##            self.b=torch.ones((1,128)).to(self.device)
##
##            
##            data_dir='D:\marimite\doujin\doujinshidata'
##            with open(os.path.join(data_dir,'merge_data.json'),'r',encoding='utf-8') as f:
##                merged_data=json.load(f)
##            l1=[]
##            self.l2=[]
##            features = []
##            
##            for i,a in enumerate(merged_data):
##                doujinshiorg_code=a['doujinshiorg_code']
##                surugaya_code=a['surugaya_code']
##                if surugaya_code:
##                    l1.append(['surugaya',surugaya_code,i,a['title']])
##                if doujinshiorg_code:
##                    l1.append(['doujinshiorg',doujinshiorg_code,i,a['title']])
##            for i in l1:
##                path=os.path.join(data_dir,i[0],'picture',i[1]+('_0' if i[0]=='doujinshiorg' else '')+'.jpg')
##                if not os.path.exists(path):
##                    continue
##                self.l2.append(i)
##                path2=os.path.join('static','img2',i[0]+'_'+i[1]+'.jpg')
##                if not os.path.exists(path2):
##                    shutil.copy(path,path2)
####                path3=os.path.join('static','feature_3_000',i[0]+'_'+i[1]+'.npy')
####                if os.path.exists(path3):
####                    feature=np.load(path3)
####                else:
####                    feature=fe.extract(img=Image.open(path2))
####                    np.save(path3, feature)
##                features.append(self.extract(path3))
##            
##            self.feature_t=torch.concat(features,dim=0)
##            self.feature_t-=self.feature_t.min(dim=0,keepdim=True)[0]
##            self.feature_t/=self.feature_t.max(dim=0,keepdim=True)[0]+0.01
##
##            torch.save({'state_dict':self.model.cpu().state_dict(),'a':self.a,'b':self.b,
##                        'feature':self.feature_t,'img_list':self.l2},model_path)
##        print(len(l1),len(self.l2))
##    def extract(self, img):
##
##        image_ = Image.open(img).resize((224, 224), Image.LANCZOS).convert('RGB')
##        image = np.asarray(image_)
##        assert len(image.shape)==3,(image.shape,image_.mode,j)
##        image = image / 255.0
##        image = torch.Tensor(image).unsqueeze_(dim=0)  # (b,h,w,c)
##        image = image.permute((0, 3, 1, 2)).float()  # (b,h,w,c) -> (b,c,h,w)
##        return (self.model(image.to(self.device)).detach()-self.a)/self.b
