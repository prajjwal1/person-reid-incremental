from torch.utils.data.dataset import Dataset
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import os
import pandas as pd
import torch

class MultiLabelDataset(Dataset):
    def __init__(self, csv_path, img_path, transform=None):
        tmp_df = pd.read_csv(csv_path)
        
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = self.mlb.fit_transform(tmp_df['tags'].str.split()).astype(np.float32)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.X_train[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)
