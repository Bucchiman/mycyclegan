#!/usr/bin/env python
# -*- coding: utf-8 -*-
# FileName: 	test
# CreatedDate:  2021-05-19 01:33:39 +0900
# LastModified: 2021-06-06 03:30:16 +0900
#


import os
import sys
import argparse
from PIL import Image
from pathlib import Path
import torch
from torchvision import transforms
import kornia
import cv2
from collections import OrderedDict
from models import GeneratorResNet


def test(img_shape,
         data_path,
         output_path,
         generator,
         device):
    generator = generator.to(device)
    generator.eval()
    mytransform = transforms.Compose([transforms.Resize(img_shape, Image.BICUBIC),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    output_transform = transforms.ToPILImage()
    for img_path in Path(data_path).iterdir():
        name = Path(img_path).name
        img = Image.open(img_path)
        green = mytransform(img)
        green = torch.unsqueeze(green, 0)
        green = green.to(device)

        output = generator(green)
        output = torch.squeeze(output)
        output = output.mul(torch.FloatTensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device))
        output = output.add(torch.FloatTensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device))
        cherry = output_transform(output)
        cherry.save(str(Path(output_path).joinpath(name)))


def main(args):
    img_shape = (args.img_height, args.img_width)
    if not Path(args.output_path).exists():
        Path(args.output_path).mkdir(parents=True)
    generator = GeneratorResNet((3, *img_shape), args.n_residual_blocks)
    weights = torch.load(args.model_path, map_location=torch.device(args.device))
    new_weights = OrderedDict()
    for key, value in weights.items():
        new_key = key.replace("module.", "")
        new_weights[new_key] = value
    generator.load_state_dict(new_weights)
    test(img_shape, args.data_path, args.output_path, generator, args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("--device", choices=("cpu", "cuda:0", "cuda:1", "cuda:2"),
                        default="cpu")
    parser.add_argument("--batch", default=1)
    parser.add_argument("--img_height", type=int, default=1024)
    parser.add_argument("--img_width", type=int, default=512)
    parser.add_argument("--n_residual_blocks", type=int, default=9)
    args = parser.parse_args()
    main(args)
