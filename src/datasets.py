import random
from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(Path(root).joinpath(mode, "A").iterdir())
        self.files_B = sorted(Path(root).joinpath(mode, "B").iterdir())

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B)-1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class TestDataset(Dataset):
    def __init__(self, path, transforms_=None):
        self.path = path
        self.img_path = [fp for fp in Path(self.path).iterdir()]
        self.transform = transforms_

    def __getitem__(self, idx):
        img_name = Path(self.img_path[idx]).name
        image = Image.open(img_name)
        image = self.transform(image)
        return {"image": image, "name": img_name}
