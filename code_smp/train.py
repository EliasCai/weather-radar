import torch
from torchsummary import summary

# from unet import UNet
import segmentation_models_pytorch as smp
import glob
import os, sys
import cv2
import numpy as np
from loss import DiceLoss
import torch.optim as optim
from dataset import WeatherRadarDataset, generate_pairs
from torch.utils.data import DataLoader
from tqdm import tqdm


def datasets():

    raw_pairs = generate_pairs("../inputs/train_data01")

    train_pairs = [p for p in raw_pairs if p[0][0].split("/")[-2][7] not in ["7"]]
    eval_pairs = [p for p in raw_pairs if p[0][0].split("/")[-2][7] in ["7"]]

    print("len of train=", len(train_pairs), "len of eval=", len(eval_pairs))
    data_train = WeatherRadarDataset(pairs=train_pairs)
    data_eval = WeatherRadarDataset(pairs=eval_pairs)

    loader_train = DataLoader(
        data_train, batch_size=4, shuffle=True, drop_last=False, num_workers=4
    )

    loader_eval = DataLoader(
        data_eval, batch_size=4, shuffle=True, drop_last=False, num_workers=4
    )

    return loader_train, loader_eval


def train_epoch(model, loader, optimizer, dsc_loss):

    model.train()
    loss_train = []
    pbar = tqdm(loader, total=len(loader))
    # pbar = loader
    for idx, data in enumerate(pbar):
        x, y_true = data
        x, y_true = x.cuda(), y_true.cuda()
        optimizer.zero_grad()

        y_pred = model(x)
        # print(x.shape,y_pred.shape)
        loss = dsc_loss(y_pred, y_true)
        loss_train.append(loss.item())
        loss.backward()
        optimizer.step()
        desc = "%.3f" % (sum(loss_train) / len(loss_train))
        pbar.set_description(desc)
        # if idx % 200 == 0:
        # print('%5d-%.3f' % (idx, (sum(loss_train) / len(loss_train))))
    return model


def eval_epoch(model, loader, dsc_loss):

    model.eval()
    loss_eval = []
    pbar = tqdm(loader, total=len(loader))
    # pbar = loader
    for idx, data in enumerate(pbar):
        x, y_true = data
        x, y_true = x.cuda(), y_true.cuda()

        y_pred = model(x)
        loss = dsc_loss(y_pred, y_true)
        loss_eval.append(loss.item())
        desc = "%.3f" % (sum(loss_eval) / len(loss_eval))
        pbar.set_description(desc)
    return model


def main():

    loss = smp.utils.losses.MSELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DEVICE = "cuda"
    metrics = [smp.utils.metrics.IoU()]
    # dsc_loss = DiceLoss()

    # model = UNet(in_channels=21, out_channels=4, init_features=128)
    model = smp.Unet(
        "resnet34",
        encoder_weights="imagenet",
        in_channels=21,
        classes=4,
        activation=None,
    )
    model.cuda()
    print(summary(model, input_size=(21, 256, 256)))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.Adam([dict(params=model.parameters(), lr=0.0001)])

    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001)
    loader_train, loader_eval = datasets()

    save_model_path = "../checkpoints/resnet34-smp.pth"
    load_model_path = "../checkpoints/resnet34-smp.pth"
    if os.path.exists(load_model_path):
        print('load model from', load_model_path)
        model = torch.load(load_model_path)
        # model.load_state_dict(torch.load(load_model_path))

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model, loss=loss, metrics=metrics, device=DEVICE, verbose=True
    )
    max_score = 0

    for i in range(0, 400):

        print("\nEpoch: {}".format(i))
        train_logs = train_epoch.run(loader_train)
        valid_logs = valid_epoch.run(loader_eval)
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model,save_model_path)
            print('Model saved!')
                              
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')


if __name__ == "__main__":
    main()
