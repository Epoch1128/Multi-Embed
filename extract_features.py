import torch
import os
import pickle as pkl
from torch.utils.data import DataLoader, Dataset
from yacs.config import CfgNode as CN
from torchvision import transforms
from glob import glob
import numpy as np
from PIL import Image
import timm
from huggingface_hub import login

login()

def _get_config():
    config = CN()

    config.data = CN()
    config.data.tile_dir = 'PATH_TO_TILES'
    config.data.save_dir = 'PATH_TO_SAVE'
    # config.data.prefix = os.listdir(config.data.tile_dir)
    config.data.prefix = ['PREFIX_A', 'PREFIX_B', 'PREFIX_C', 'PREFIX_D']
    return config

def load_image(img_path):
    fp = open(img_path, 'rb')
    pic = Image.open(fp)
    pic = np.array(pic)
    fp.close()
    return pic
        
class TilesDataset(Dataset):
    def __init__(self, tile_dir, prefix=['*']):
        self.tile_list = []
        for dir_prefix in prefix:
            self.tile_list.extend(glob(os.path.join(tile_dir, f"{dir_prefix}/*.jpg")))
        self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )

    def __len__(self):
        return len(self.tile_list)
    
    def __getitem__(self, index):
        img_path = self.tile_list[index]
        img_name = img_path.split('/')[-1].split('.')[0]
        sample_name = img_path.split('/')[-2]
        pic = load_image(img_path)
        image = self.transforms(pic)
        return {
            'name': f'{sample_name}_{img_name}',
            'image': image
        }

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = _get_config()
    
    print("Load Model...")
    
    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using 1 gpu")
    else:
        print("Using cpu")

    model.to(device)

    for prefix in config.data.prefix:
        val_data = TilesDataset(config.data.tile_dir, prefix=[prefix])
        print(f"length of val data: {len(val_data)}")
        val_loader = DataLoader(val_data, batch_size=2)

        val_results = val_loop(model, val_loader, device)
        os.makedirs(config.data.save_dir, exist_ok=True)
        with open(os.path.join(config.data.save_dir, f'{prefix}.pkl'), 'wb') as file:
            pkl.dump(val_results, file)
                
    print("Val Finished!")


def val_loop(model, val_loader, device):
    model.eval()
    
    all_results = {}
    for idx, data in enumerate(val_loader):
        images = data['image'].to(torch.float32).to(device)
        feats = model(images)

        for name, feat in zip(data['name'], feats.detach().cpu().numpy()):
            all_results.update(
                {
                    name: feat
                }
            )

    return all_results


if __name__ == '__main__':
    main()