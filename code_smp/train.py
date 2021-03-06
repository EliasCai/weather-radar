import segmentation_models_pytorch as smp
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
from sklearn.metrics import accuracy_score


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


def cal_acc(y_true, y_pred):
    y_true = y_true.view(-1).detach().cpu().numpy()
    y_pred = y_pred.view(-1).detach().cpu().numpy()
    return accuracy_score(
        np.floor(y_true * 80 / 10).astype(np.int),
        np.floor(y_pred * 80 / 10).astype(np.int),
    )


def train_epoch(model, loader, optimizer, dsc_loss):

    model.train()
    loss_train = []
    acc_train = []
    pbar = tqdm(loader, total=len(loader))
    # pbar = loader
    loss_func = torch.nn.BCELoss()
    for idx, data in enumerate(pbar):
        x, y_true = data
        x, y_true = x.cuda(), y_true[:, :, :, :].cuda()
        optimizer.zero_grad()

        y_pred = model(x)
        # print(x.shape,y_pred.shape)
        loss = dsc_loss(y_pred, y_true) + loss_func(
            (y_pred > 0).view(-1).float(), (y_true > 0).view(-1).float()
        )
        loss_train.append(loss.item())
        acc_train.append(cal_acc(y_true, y_pred))
        loss.backward()
        optimizer.step()
        desc = "loss - %.3f acc - %.3f" % (
            sum(loss_train) / len(loss_train),
            sum(acc_train) / len(acc_train),
        )

        pbar.set_description(desc)
        # if idx % 200 == 0:
        # print('%5d-%.3f' % (idx, (sum(loss_train) / len(loss_train))))
    return model


def eval_epoch(model, loader, dsc_loss):

    model.eval()
    loss_eval = []
    acc_eval = []
    pbar = tqdm(loader, total=len(loader))
    # pbar = loader
    loss_func = torch.nn.BCELoss()
    for idx, data in enumerate(pbar):
        x, y_true = data
        x, y_true = x.cuda(), y_true[:, :, :, :].cuda()

        y_pred = model(x)
        # loss = dsc_loss(y_pred, y_true)
        loss = dsc_loss(y_pred, y_true) + loss_func(
            (y_pred > 0).view(-1).float(), (y_true > 0).view(-1).float()
        )
        loss_eval.append(loss.item())
        acc_eval.append(cal_acc(y_true, y_pred))
        desc = "loss - %.3f acc - %.3f" % (
            sum(loss_eval) / len(loss_eval),
            sum(acc_eval) / len(acc_eval),
        )
        pbar.set_description(desc)
    return model


def main():

    dsc_loss = smp.utils.losses.MSELoss(reduction="mean")
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
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01)
    loader_train, loader_eval = datasets()

    save_model_path = "../checkpoints/resnet34-all.pk"
    load_model_path = "../checkpoints/resnet34-all.pk"
    if os.path.exists(load_model_path):
        print("restore model from", load_model_path)
        model.load_state_dict(torch.load(load_model_path))

    for e in range(1000):
        model = train_epoch(model, loader_train, optimizer, dsc_loss)
        model = eval_epoch(model, loader_eval, dsc_loss)
        # scheduler.step()
        torch.save(model.state_dict(), save_model_path)
        print("begin epoch", e, "saving to", save_model_path)  # ,scheduler.get_lr())


if __name__ == "__main__":
    main()
