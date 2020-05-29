import os
import random
import glob
import numpy as np
import torch

# from skimage.io import imread
import cv2
from torch.utils.data import Dataset

# from utils import crop_sample, pad_sample, resize_sample, normalize_volume


def generate_pairs(search_path="../data/huawei/train/"):

    # assert period in ['30min','60min','90min','120min']

    pairs = []
    folders = os.listdir(search_path)
    for folder in folders:
        image_paths = glob.glob(os.path.join(search_path, folder, "*.png"))
        image_paths = sorted(image_paths)
        # print(image_paths)
        # assert len(image_paths) == 41 or len(image_paths) == 21
        if (len(image_paths) != 41) and (len(image_paths) != 21):
            continue
        x = image_paths[:21]
        if len(image_paths) == 41:
            y_30 = image_paths[25]
            y_60 = image_paths[30]
            y_90 = image_paths[35]
            y_120 = image_paths[40]
            pairs.append((x, (y_30, y_60, y_90, y_120)))
        elif len(image_paths) == 21:
            pairs.append((x, ("", "", "", "")))
    return pairs


class WeatherRadarDataset(Dataset):

    in_channels = 21
    out_channels = 4

    def __init__(self, pairs, transform=None):
        self.transform = transform
        self.pairs = pairs

    def __images_to_np(self, image_paths):

        imgs = []
        for image_path in image_paths[:21]:
            img = cv2.imread(image_path)
            imgs.append(img[:, :, 0])
        imgs = np.stack(imgs, 2)
        imgs = np.transpose(imgs, (2, 0, 1))
        # imgs = np.expand_dims(imgs,0)
        imgs = imgs.astype(np.float32) / 80

        return imgs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        x_paths, (y30_path, y60_path, y90_path, y120_path) = self.pairs[idx]

        x_np = self.__images_to_np(x_paths)
        image_tensor = torch.from_numpy(x_np)
        if y30_path == y60_path == y90_path == y120_path == '':
            return image_tensor, torch.from_numpy(np.array([idx]))
        y_np = self.__images_to_np([y30_path, y60_path, y90_path, y120_path])

        # if self.transform is not None:
        #     image, mask = self.transform((image, mask))

        mask_tensor = torch.from_numpy(y_np)

        return image_tensor, mask_tensor


if __name__ == "__main__":
    train_pairs = generate_pairs("../inputs/train_data01/")
    test_pairs = generate_pairs("../inputs/test_data/")
    print(test_pairs)
