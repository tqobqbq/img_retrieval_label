import torch
import numpy as np
import sys
sys.path.append(r'D:\code\SimCLR-master\models')
from resnet_simclr import ResNetSimCLR
# from torchvision.transforms import transforms
import torchvision.transforms.v2 as transforms
import torch
from gaussian_blur import GaussianBlur
# See https://keras.io/api/applications/ for details
PATCH_SIZE = 16
IMAGE_SIZE = 768

# quantization filter for the given patch size
patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))

# image resize transform to dimensions divisible by patch size
def resize_transform(
    mask_image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))
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
        patch_mask_values = []
        patch_features = []
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                for image, mask in tqdm([(image_left, mask_left), (image_right, mask_right)], desc="Processing images"):
                    # processing mask
                    mask = mask.split()[-1]
                    mask_resized = resize_transform(mask)
                    #mask_quantized = patch_quant_filter(mask_resized).squeeze().view(-1).detach().cpu()
                    mask_quantized = patch_quant_filter(mask_resized).squeeze().detach().cpu()
                    patch_mask_values.append(mask_quantized)
                    # processing image
                    image = image.convert('RGB')
                    image_resized = resize_transform(image)
                    image_resized = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    image_resized = image_resized.unsqueeze(0).cuda()

                    feats = model.get_intermediate_layers(image_resized, n=range(n_layers), reshape=True, norm=True)
                    dim = feats[-1].shape[1]
                    #patch_features.append(feats[-1].squeeze().view(dim, -1).permute(1,0).detach().cpu())
                    patch_features.append(feats[-1].squeeze().detach().cpu())



class FeatureExtractor2:
    def __init__(self,path,device):
##        base_model = VGG16(weights='imagenet')
##        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        s=1
        color_jitter =transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        size=96
##        self.transform=transforms.Compose([transforms.Resize(size=size),
####                                              transforms.RandomResizedCrop(size=size),
####                                              transforms.RandomHorizontalFlip(),
####                                              transforms.RandomApply([color_jitter], p=0.8),
####                                              transforms.RandomGrayscale(p=0.2),
##                                              GaussianBlur(kernel_size=int(0.1 * size)),
##                                              transforms.ToTensor()])
##        self.transform = transforms.Compose([
####                                            transforms.RandomRotation(180),
####                                            transforms.RandomResizedCrop(scale=(0.8,1.5),ratio=(0.8,1.25),size=size),
##                                            
####                                              transforms.RandomHorizontalFlip(),
####                                              transforms.RandomApply([color_jitter], p=0.8),
####                                              transforms.RandomGrayscale(p=0.2),
####                                              GaussianBlur(kernel_size=int(0.1 * size)),
##                                              transforms.Resize(size),
##                                              transforms.ToTensor()])
        self.transform=transforms.Compose([
                                    transforms.Resize(size=(size,size)),
                                    # transforms.RandomResizedCrop(scale=(0.8,1.5),ratio=(0.8,1.25),size=size),
                                    transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
                                    ])
        self.model=ResNetSimCLR('resnet18',128)
        self.model.load_state_dict(torch.load(path)['state_dict'])
        self.device=device
        self.model=self.model.to(self.device)
        self.model.eval()

    def extract(self, img):
#         input=self.transform(img.convert('RGB'))
# ##        if input.shape==1:
# ##
#         with torch.no_grad():
#             output=torch.nn.functional.normalize(self.model(input.unsqueeze(0)),dim=1)[0]
#         return output.detach().numpy()

        input = self.transform(img).unsqueeze(0).to(self.device)
    # with torch.autocast('cuda', dtype=torch.bfloat16):
    #     batch_img = transform(img)[None]
    #     batch_img = batch_img
    #     depths = depther(batch_img)
        with torch.no_grad():
            # output = torch.nn.functional.normalize(self.model(input), dim=1)[0]
            # output = torch.nn.functional.normalize(self.model(input), dim=1)[0]
            output = self.model(input)[0]
        return output

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


def make_transform(resize_size: int | list[int] = 768):
    to_tensor = transforms.ToImage()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    to_float = transforms.ToDtype(torch.float32, scale=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, to_float, normalize])
class FeatureExtractor4:
    def __init__(self,device):
        # self.model=dinov3_convnext_large = torch.hub.load("facebookresearch/dinov3", 'dinov3_convnext_large', source='github', weights='dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth').to(device)
        self.device=device  
        self.load_model()
        img_size = 224
        self.transform=make_transform(img_size)
        self.model.eval()

    def load_model(self,):
        self.model=dinov3_convnext_large = torch.hub.load(r"C:\Users\Administrator\.cache\torch\hub\facebookresearch_dinov3_main", 'dinov3_convnext_large', source='local', weights='dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth').to(self.device)

    
    def extract(self, img):
        input = self.transform(img).unsqueeze(0).to(self.device)
    # with torch.autocast('cuda', dtype=torch.bfloat16):
    #     batch_img = transform(img)[None]
    #     batch_img = batch_img
    #     depths = depther(batch_img)
        with torch.no_grad():
            # output = torch.nn.functional.normalize(self.model(input), dim=1)[0]
            # output = torch.nn.functional.normalize(self.model(input), dim=1)[0]
            output = self.model(input)[0]
        return output
    
class FeatureExtractor5(FeatureExtractor4):
    def load_model(self,):
        # self.model= torch.hub.load("facebookresearch/dinov2", 'dinov2_vitl14', source='github', weights='dinov2_vitl14_pretrain.pth').to(self.device)
        self.model= torch.hub.load(r"C:\Users\Administrator\.cache\torch\hub\facebookresearch_dinov2_main", 'dinov2_vitl14', source='local', weights='dinov2_vitl14_pretrain.pth').to(self.device)
    
class FeatureExtractor6(FeatureExtractor4):
    def load_model(self,):
        self.model= torch.hub.load(r"C:\Users\Administrator\.cache\torch\hub\facebookresearch_dinov3_main", 'dinov3_vith16plus', source='local', weights='dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth').to(self.device)
if __name__ == '__main__':
    fe1=FeatureExtractor4(device='cpu')
    
    # print(fe1.model)
    print(fe1.extract(np.zeros((224,224,3),dtype=np.uint8)).shape)
    # print(fe1.extract(torch.zeros((224,224,3),dtype=torch.uint8)).shape)