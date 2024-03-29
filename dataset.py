import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class CarvanaDataset(Dataset):

    def __init__(self, image_dir, mask_dir, trasform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = trasform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return(len(self.images))

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        #some preprocessing
        mask[mask == 225.0] = 1

        if self.transform is not None:
            augmentations = self.transform(image=image,mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image,mask
