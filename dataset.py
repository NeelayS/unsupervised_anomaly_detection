from torch.utils.data import Dataset
from torchvision import transforms
import torch
import cv2 as cv
import os
import pandas as pd
import random


class RNSADataset(Dataset):
    def __init__(self, root_dir, img_list):
        self.root_dir = root_dir
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_list[idx] + ".jpg")

        try:
            img = cv.imread(img_path)
        except:
            pass

        img = cv.resize(img, dsize=(256, 256), interpolation=cv.INTER_LANCZOS4)

        return img


def make_dataset(root_dir, class_info_csv, sampling_ratio=0.7):

    class_info = pd.read_csv(class_info_csv)

    normal = list(class_info[class_info["class"] == "Normal"]["patientId"])
    random.shuffle(normal)

    opaque = list(class_info[class_info["class"] == "Lung Opacity"]["patientId"])
    random.shuffle(opaque)

    other_abnormal = list(
        class_info[class_info["class"] == "No Lung Opacity / Not Normal"]["patientId"]
    )
    random.shuffle(other_abnormal)

    n_train = int(len(normal) * sampling_ratio) + 1
    train_list = normal[:n_train]
    leftover_normal = normal[n_train:]

    n_leftover_normal = len(leftover_normal)
    n_opaque = len(opaque)
    n_other_abnormal = len(other_abnormal)

    val_list = (
        leftover_normal[: n_leftover_normal // 3]
        + opaque[: n_opaque // 3]
        + other_abnormal[: n_other_abnormal // 3]
    )
    test_list = (
        leftover_normal[n_leftover_normal // 3 :]
        + opaque[n_opaque // 3 :]
        + other_abnormal[n_other_abnormal // 3 :]
    )

    train_dataset = RNSADataset(root_dir, train_list)
    val_dataset = RNSADataset(root_dir, val_list)
    test_dataset = RNSADataset(root_dir, test_list)

    return (train_dataset, val_dataset, test_dataset)
