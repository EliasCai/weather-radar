import segmentation_models_pytorch as smp
import torch
from torchsummary import summary
import argparse
# from unet import UNet
import segmentation_models_pytorch as smp
import glob
import os, sys
import cv2
import numpy as np
from loss import DiceLoss
import torch.optim as optim
from dataset_classfication import WeatherRadarDataset, generate_pairs
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from augmentation import get_training_augmentation


def datasets():

    raw_pairs = generate_pairs("../inputs/train_data01") + \
            generate_pairs("../inputs/train_data02")
    raw_reverse_pairs = generate_pairs("../inputs/train_data01", True) + \
            generate_pairs("../inputs/train_data02", True)

    train_pairs = [p for p in raw_pairs if p[0][0].split("/")[-2][7] not in ["7"]] + [
        p for p in raw_reverse_pairs if p[0][0].split("/")[-2][7] not in ["7"]
    ]
    eval_pairs = [p for p in raw_pairs if p[0][0].split("/")[-2][7] in ["7"]]

    print("len of train=", len(train_pairs), "len of eval=", len(eval_pairs))
    # data_train = WeatherRadarDataset(pairs=train_pairs, augmentation=get_training_augmentation())
    data_train = WeatherRadarDataset(pairs=train_pairs)
    data_eval = WeatherRadarDataset(pairs=eval_pairs)

    loader_train = DataLoader(
        data_train, batch_size=8, shuffle=True, drop_last=False, num_workers=4
    )

    loader_eval = DataLoader(
        data_eval, batch_size=8, shuffle=True, drop_last=False, num_workers=4
    )

    return loader_train, loader_eval


def cal_metrics(y_true, y_pred):
    y_true = y_true.view(-1).detach().cpu().numpy()
    y_pred = y_pred.argmax(1).view(-1).detach().cpu().numpy()
    # return accuracy_score(
    #     np.floor(y_true / 0.1).astype(np.int), np.floor(y_pred / 0.1).astype(np.int)
    # )
    return f1_score(y_true, y_pred, average="macro")


def train_epoch(model, loader, optimizer, loss_func, tidx):

    model.train()
    loss_train = []
    acc_train = []
    pbar = tqdm(loader, total=len(loader))
    # pbar = loader
    # loss_func2=torch.nn.BCELoss()
    for idx, data in enumerate(pbar):
        x, y_true = data
        x, y_true = x.cuda(), y_true[:, tidx, :, :].cuda()
        optimizer.zero_grad()

        y_pred = model(x)
        # print(x.shape,y_pred.shape)
        loss = loss_func(y_pred, y_true)
        loss_train.append(loss.item())
        acc_train.append(cal_metrics(y_true, y_pred))
        loss.backward()
        optimizer.step()
        desc = "loss - %.3f acc - %.3f" % (
            sum(loss_train) / len(loss_train),
            sum(acc_train) / len(acc_train),
        )

        pbar.set_description(desc)
    return model


def eval_epoch(model, loader, loss_func, tidx):

    model.eval()
    loss_eval = []
    acc_eval = []
    pbar = tqdm(loader, total=len(loader))
    # pbar = loader
    # loss_func=torch.nn.BCELoss()
    for idx, data in enumerate(pbar):
        x, y_true = data
        x, y_true = x.cuda(), y_true[:, tidx, :, :].cuda()

        y_pred = model(x)
        # loss = dsc_loss(y_pred, y_true)
        loss = loss_func(y_pred, y_true)
        loss_eval.append(loss.item())
        acc_eval.append(cal_metrics(y_true, y_pred))
        desc = "loss - %.3f acc - %.3f" % (
            sum(loss_eval) / len(loss_eval),
            sum(acc_eval) / len(acc_eval),
        )
        pbar.set_description(desc)
    return model, sum(acc_eval) / len(acc_eval)


def main(time_stamp):

    time_to_idx = {"30": 0, "60": 1, "90": 2, "120": 3}
    tidx = time_to_idx[time_stamp]
    loss_func = smp.utils.losses.CrossEntropyLoss()
    # loss_func = smp.utils.losses.DiceLoss()

    model = smp.Unet(
        "resnet34",
        encoder_weights="imagenet",
        in_channels=21,
        classes=5,
        activation=None,
    )
    model.cuda()
    print(summary(model, input_size=(21, 256, 256)))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01)
    loader_train, loader_eval = datasets()

    save_model_path = "../checkpoints/resnet34-min%s.pk" % time_stamp
    load_model_path = "../checkpoints/resnet34-min%s.pk" % time_stamp
    if os.path.exists(load_model_path):
        print("restore model from", load_model_path)
        model.load_state_dict(torch.load(load_model_path))

    max_score = 0.1
    stop_learning = 0
    for e in range(100):
        model = train_epoch(model, loader_train, optimizer, loss_func, tidx)
        model, score = eval_epoch(model, loader_eval, loss_func, tidx)
        # scheduler.step()
        if score > max_score:
            torch.save(model.state_dict(), save_model_path)
            print(
                "epoch-%d" % e, "saving to", save_model_path, score
            )  # ,scheduler.get_lr())
            max_score = score
        else:
            stop_learning += 1
        if stop_learning > 4:
            break


def parse_args():
    """ 
    Parse command line arguments.    
    """
    parser = argparse.ArgumentParser(description="Weather Radar")
    parser.add_argument("--time_stamp", help="Time stamp", default=None, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.time_stamp)
