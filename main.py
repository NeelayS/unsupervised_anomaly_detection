import argparse
import torch.nn.functional as F
from torch.optim import Adam
import torch

from torch.utils.data import DataLoader
from datetime import datetime

from dataset import make_dataset
from train import train
from model import AnomalyAE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',
                        required=True,
                        help="Please specify the train directory")
    parser.add_argument('--class_info_csv',
                        required=True,
                        help="Please specify path to the class info csv file")
    parser.add_argument(
        "--log_interval", type=int, default=10,
        help="how many batches to wait before logging training status"
    )
    parser.add_argument('--epochs',
                        type=int,
                        default=25,
                        help="Please specify the number of epochs")
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=4,
                        help="Please specify the batch_size")
    parser.add_argument('--val_batch_size',
                        type=int,
                        default=4,
                        help="Please specify the batch_size")

    parser.add_argument(
        "--log_dir", type=str,
        default=f'tensorboard_logs_{datetime.now().strftime("%d%m%Y_%H-%M")}',
        help="log directory for Tensorboard log output"
    )
    parser.add_argument(
        '--load_weight_path', type=str,
        help="Please specify the weight path that needs to be loaded.")

    parser.add_argument(
        '--save_plot', action='store_true',
        help="Specify this if you want to save the network graph.")

    args = parser.parse_args()

    optimizer = Adam
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = F.mse_loss

    train_dataset, val_dataset, test_dataset = make_dataset(args.rootdir, args.class_info_csv)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle=False)
    model = AnomalyAE()

    train(model, optimizer, loss, train_loader,
          val_loader, args.log_dir, device, args.epochs,
          args.log_interval,
          args.load_weight_path, args.save_plot)