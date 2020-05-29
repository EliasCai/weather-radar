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


def generate_image(folders, y_preds):

    y_preds = (y_preds * 80).astype(np.uint8)
    for idx, (folder, y_pred) in enumerate(zip(folders, y_preds)):
        save_folder = os.path.join("../outputs/Predict", folder)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        img30 = y_preds[idx, :, :, 0]
        img60 = y_preds[idx, :, :, 1]
        img90 = y_preds[idx, :, :, 2]
        img120 = y_preds[idx, :, :, 3]
        cv2.imwrite(os.path.join(save_folder, "30.png"), img30)
        cv2.imwrite(os.path.join(save_folder, "60.png"), img60)
        cv2.imwrite(os.path.join(save_folder, "90.png"), img90)
        cv2.imwrite(os.path.join(save_folder, "120.png"), img120)


def infer(model, loader, pairs):

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
            y_pred = y_pred.detach().cpu().numpy()
            y_pred = np.transpose(y_pred, (0, 2, 3, 1))
            # print(y_pred.shape)
        generate_image(folders, y_pred)
    return model


def main():

    model = smp.Unet(
        "resnet34",
        encoder_weights="imagenet",
        in_channels=21,
        classes=4,
        activation=None,
    )
    # model = UNet(in_channels=21, out_channels=4, init_features=128)
    model.cuda()
    # print(summary(model, input_size=(21, 256, 256)))
    loader_test, test_pairs = datasets()

    load_model_path = "../checkpoints/resnet34-all.pk"
    assert os.path.exists(load_model_path) == True
    model.load_state_dict(torch.load(load_model_path))
    # model = torch.load(load_model_path)
    infer(model, loader_test, test_pairs)


if __name__ == "__main__":
    main()
