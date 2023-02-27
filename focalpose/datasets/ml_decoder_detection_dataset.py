import torchvision
import torch
import numpy as np
from PIL import Image

class MLDecoderDetectionDataset(torchvision.datasets.VisionDataset):
    def __init__(self, scene_ds, label_to_category_id, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.scene_ds = scene_ds
        self.label_to_category_id = label_to_category_id
        self.n_classes = len(label_to_category_id)

    def __getitem__(self, index):
        rgb,mask,state = self.scene_ds[index]

        output = torch.zeros((3, self.n_classes), dtype=torch.long)
        for obj in state['objects']:
            area = np.sqrt(np.sum((obj['bbox'][2:4] - obj['bbox'][:2])**2))
            if area < 32 * 32:
                output[0][self.label_to_category_id[obj['name']]] = 1
            elif area < 96 * 96:
                output[1][self.label_to_category_id[obj['name']]] = 1
            else:
                output[2][self.label_to_category_id[obj['name']]] = 1
            

        target = output
        if isinstance(rgb, torch.Tensor):
            img = Image.fromarray(rgb.numpy())
        else:
            img = Image.fromarray(rgb)
            
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    def __len__(self):
        return len(self.scene_ds)