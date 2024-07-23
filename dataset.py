import os
import cv2
import glob
import pandas as pd

from torch.utils.data import Dataset

class BrainMriDataset(Dataset):
    DATA_PATH = "./brain_mri"

    def __init__(self, transform = None, image_transform = None, mask_transform = None):
        self.dataset = self.load_data()

        self.transform = transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def response_dist(self):
        diagnosis = self.dataset['diagnosis'].value_counts()
        total = diagnosis.sum()
        cdf = {
            0: diagnosis.iloc[0] / total,
            1: diagnosis.iloc[1] / total
        }

        return cdf

    def load_data(self):
        image_to_mask = {}

        for sub_folder in glob.glob(BrainMriDataset.DATA_PATH + "/*"):
            for file in glob.glob(sub_folder + "/*"):
                if "mask" not in file:
                    continue
                
                mask_path = file
                image_path = file.replace("_mask", "")
                if not os.path.exists(image_path):
                    print("Orignal image not found for mask", mask_path.split('/')[-1])
                    continue
                image_to_mask[image_path] = mask_path

        file, images, masks = [], [], []
        for image, mask in image_to_mask.items():
            file.append(image.split("/")[-1])
            images.append(cv2.imread(image))
            masks.append(cv2.imread(mask))

        df = pd.DataFrame({"file": file, "image": images, "mask": masks})
        df["diagnosis"] = df["mask"].apply(lambda mask : 1 if mask.max() > 0 else 0)

        return df
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x = self.dataset.iloc[idx].copy()
        img, mask = x["image"], x["mask"]

        if self.transform:
            a = self.transform(img, mask)
            img, mask = a
        if self.image_transform:
            img = self.image_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return {"image": img, "mask": mask, "diagnosis": x["diagnosis"]}