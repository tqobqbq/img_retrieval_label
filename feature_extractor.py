import torch
import numpy as np
import sys
sys.path.append(r'D:\code\SimCLR-master\models')
from resnet_simclr import ResNetSimCLR
# from torchvision.transforms import transforms
import torchvision.transforms.v2 as transforms
# import torchvision.transforms.functional as F
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
        
    @torch.inference_mode()
    def extract(self, img,rotate=False):
        if rotate:
            transformed_img = self.transform(img)
            rotates=[transformed_img]
            for _ in range(3):
                rotates.append(transforms.functional.rotate(rotates[-1], 90))
            input = torch.stack(rotates).to(self.device)
        else:
            input = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input)
        return output


def make_transform(resize_size: int | list[int] = 768):
    to_tensor = transforms.ToImage()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    to_float = transforms.ToDtype(torch.float32, scale=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, to_float, normalize])
class FeatureExtractor4(FeatureExtractor2):
    def __init__(self,device):
        # self.model=dinov3_convnext_large = torch.hub.load("facebookresearch/dinov3", 'dinov3_convnext_large', source='github', weights='dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth').to(device)
        self.device=device  
        self.load_model()
        img_size = 224
        self.transform=make_transform(img_size)
        self.model.eval()

    def load_model(self,):
        self.model=dinov3_convnext_large = torch.hub.load(r"C:\Users\Administrator\.cache\torch\hub\facebookresearch_dinov3_main", 'dinov3_convnext_large', source='local', weights='dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth').to(self.device)

    
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