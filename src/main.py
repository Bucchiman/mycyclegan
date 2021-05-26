#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	main
# CreatedDate:  2021-04-30 20:14:48 +0900
# LastModified: 2021-05-18 22:06:23 +0900
#


from datetime import datetime
import argparse
from itertools import chain
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from pathlib import Path
from models import GeneratorResNet, Discriminator, weights_init_normal
from utils import LambdaLR, ReplayBuffer
from datasets import ImageDataset
from train import train


def main(args):
    time_keeper = datetime.now().strftime(r"%Y_%m_%d_%H_%M")
    args.output_path = str(Path(args.output_path).joinpath(time_keeper))
    Path(args.output_path).mkdir(parents=True)
    saved_models_path = str(Path(args.output_path).joinpath("saved_models"))
    Path(saved_models_path).mkdir()

    criterion_GAN = nn.MSELoss()
    criterion_GAN = criterion_GAN.to(args.device)
    criterion_cycle = nn.L1Loss()
    criterion_cycle = criterion_cycle.to(args.device)
    criterion_identity = nn.L1Loss()
    criterion_identity = criterion_identity.to(args.device)

    input_shape = (args.channels, args.img_height, args.img_width)
    G_AB = GeneratorResNet(input_shape, args.n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, args.n_residual_blocks)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    optimizer_G = optim.Adam(chain(G_AB.parameters(),
                                   G_BA.parameters()),
                             lr=args.lr,
                             betas=(args.b1, args.b2))
    optimizer_D_A = optim.Adam(D_A.parameters(),
                               lr=args.lr,
                               betas=(args.b1, args.b2))
    optimizer_D_B = optim.Adam(D_B.parameters(),
                               lr=args.lr,
                               betas=(args.b1, args.b2))
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G,
                                                 lr_lambda=LambdaLR(args.n_epochs,
-                                                                   0,
-                                                                   args.decay_epoch).step)
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                   lr_lambda=LambdaLR(args.n_epochs,
-                                                                     0,
-                                                                     args.decay_epoch).step)
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                   lr_lambda=LambdaLR(args.n_epochs,
                                                                      0,
                                                                      args.decay_epoch).step)
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    transforms_ = [transforms.Resize((args.img_height,
                                      args.img_width),
                                     Image.BICUBIC),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    train_dataset = ImageDataset(str(Path(args.data_path).joinpath(args.dataset_name)),
                                 transforms_=transforms_,
                                 unaligned=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.n_cpu)
    valid_dataset = ImageDataset(str(Path(args.data_path).joinpath(args.dataset_name)),
                                 transforms_=transforms_,
                                 unaligned=True,
                                 mode="test")
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=5,
                                  shuffle=True,
                                  num_workers=1)
    train(args.output_path,
          args.dataset_name,
          args.n_epochs,
          D_A,
          D_B,
          G_AB,
          G_BA,
          train_dataloader,
          valid_dataloader,
          optimizer_G,
          optimizer_D_A,
          optimizer_D_B,
          criterion_GAN,
          criterion_identity,
          criterion_cycle,
          args.lambda_id,
          args.lambda_cyc,
          fake_A_buffer,
          fake_B_buffer,
          args.device,
          lr_scheduler_G,
          lr_scheduler_D_A,
          lr_scheduler_D_B,
          args.epoch_interval,
          args.checkpoint_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("output_path")
    parser.add_argument("--device",
                        type=str,
                        choices=["cpu", "cuda:0", "cuda:1"],
                        default="cpu")
    parser.add_argument("--n_epochs",
                        type=int,
                        default=200,
                        help="number of epochs of training")
    parser.add_argument("--dataset_name",
                        type=str,
                        choices=["monet2photo", "mycustom", "sample", "web"],
                        default="sample",
                        help="name of the dataset")
    parser.add_argument("--batch_size",
                        type=int,
                        default=10,
                        help="size of the batches")
    parser.add_argument("--lr",
                        type=float,
                        default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1",
                        type=float,
                        default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2",
                        type=float,
                        default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch",
                        type=int,
                        default=100,
                        help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu",
                        type=int,
                        default=8,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height",
                        type=int,
                        default=256,
                        help="size of image height")
    parser.add_argument("--img_width",
                        type=int,
                        default=256,
                        help="size of image width")
    parser.add_argument("--channels",
                        type=int,
                        default=3,
                        help="number of image channels")
    parser.add_argument("--epoch_interval",
                        type=int,
                        default=50,
                        help="interval between saving generator outputs")
    parser.add_argument("--checkpoint_interval",
                        type=int,
                        default=50,
                        help="interval between saving model checkpoints")
    parser.add_argument("--n_residual_blocks",
                        type=int,
                        default=9,
                        help="number of residual blocks in generator")
    parser.add_argument("--lambda_cyc",
                        type=float,
                        default=10.0,
                        help="cycle loss weight")
    parser.add_argument("--lambda_id",
                        type=float,
                        default=5.0,
                        help="identity loss weight")
    args = parser.parse_args()
    main(args)
