import torch
from torchsummary import summary
from unet import UNet
import glob
import os, sys
import cv2
import numpy as np
from loss import DiceLoss
import torch.optim as optim
from dataset import WeatherRadarDataset, generate_pairs
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp


def datasets():

    test_pairs = generate_pairs("../inputs/test_data/TEST1/")

    data_test = WeatherRadarDataset(pairs=test_pairs)

    loader_test = DataLoader(data_test, batch_size=4, drop_last=False, num_workers=4)

    return loader_test, test_pairs


def generate_image(folders, y_preds, time_stamp):

    imgs = np.where(y_preds == 1,15,0) + np.where(y_preds == 2,25,0) + np.where(y_preds == 3,35,0) + np.where(y_preds == 4,75,0)
    imgs = imgs.astype(np.uint8)
    for idx, (folder, y_pred) in enumerate(zip(folders, y_preds)):
        save_folder = os.path.join("../outputs/Predict", folder)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if time_stamp == '30':
            img30 = imgs[idx, :, :]
            cv2.imwrite(os.path.join(save_folder, "30.png"), img30)
        elif time_stamp == '60':
            img60 = imgs[idx, :, :]
            cv2.imwrite(os.path.join(save_folder, "60.png"), img60)
        elif time_stamp == '90':
            img90 = imgs[idx, :, :]
            cv2.imwrite(os.path.join(save_folder, "90.png"), img90)
        elif time_stamp == '120':
            img120 = imgs[idx, :, :]
            cv2.imwrite(os.path.join(save_folder, "120.png"), img120)


def infer(model, loader, pairs, time_stamp):

    model.eval()
    pbar = tqdm(loader, total=len(loader))
    # pbar = loader
    for i, data in enumerate(pbar):
        x, idx = data
        x = x.cuda()
        idx = idx.detach().cpu().numpy()
        folders = [pairs[k][0][0].split("/")[-2] for k in np.squeeze(idx)]
        with torch.no_grad():
            y_pred = model(x)
            y_pred = y_pred.argmax(1)
            y_pred = y_pred.detach().cpu().numpy()
        generate_image(folders, y_pred, time_stamp)
    return model


def main(time_stamp):

    time_to_idx = {"30": 0, "60": 1, "90": 2, "120": 3}
    tidx = time_to_idx[time_stamp]
    model = smp.Unet(
        "resnet34",
        encoder_weights="imagenet",
        in_channels=21,
        classes=5,
        activation=None,
    )
    model.cuda()
    loader_test, test_pairs = datasets()

    load_model_path = "../checkpoints/resnet34-min%s.pk" % time_stamp
    assert os.path.exists(load_model_path) == True
    model.load_state_dict(torch.load(load_model_path))
    # model = torch.load(load_model_path)
    infer(model, loader_test, test_pairs, time_stamp)


if __name__ == "__main__":
    for time_stamp in ['30','60','90','120']:
        main(time_stamp)
