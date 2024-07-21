import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
import argparse
import torch
from torch.nn.functional import upsample
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def load_images(directory_path, target_shape):
    images = []
    cnt = 0
    target_shape = (target_shape, target_shape, target_shape)
    for filename in sorted(os.listdir(directory_path)):
        print(f"importing {filename}")
        if cnt == 2:
            break
        if filename.endswith(".tif"):
            image = tiff.imread(os.path.join(directory_path, filename))
            if image.shape != target_shape:
                image = resize(
                    image, target_shape, preserve_range=True, anti_aliasing=True
                )
                image = image.astype(np.float32)
            images.append(image)
        cnt += 1
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)  # Add channel dimension
    return images


class CustomDataset(Dataset):
    def __init__(self, source_images, downsampling_factor):
        self.source_images = torch.tensor(source_images, dtype=torch.float32)
        print(f"before permutation shape {self.source_images.shape}")
        self.source_images = self.source_images.permute(0, 4, 2, 3, 1)
        print(f"after permutation shape {self.source_images.shape}")
        self.source_images = torch.nn.functional.interpolate(
            self.source_images, scale_factor=1 / downsampling_factor
        )
        self.source_images = self.source_images.permute(0, 2, 3, 4, 1)
        print(f"after interpolation shape {self.source_images.shape}")

    def __len__(self):
        return len(self.source_images)

    def shape(self):
        return self.source_images.shape

    def __getitem__(self, idx):
        source = self.source_images[idx]
        return source


def downsample_permute(image, downsampling_factor):
    image = torch.tensor(image, dtype=torch.float32)
    image = image.permute(0, 4, 2, 3, 1)
    print(f"first permutation shape {image.shape}")
    image = torch.nn.functional.interpolate(image, scale_factor=1 / downsampling_factor)
    image = image.permute(0, 2, 3, 4, 1)
    print(f"second permutation shape {image.shape}")
    return image


def upsample_permute(image, upsampling_factor):
    image = torch.tensor(image, dtype=torch.float32)
    print(f"image shape {image.shape}")
    image = image.permute(0, 4, 2, 3, 1)
    print(f"first permutation shape {image.shape}")
    image = torch.nn.functional.interpolate(image, scale_factor=upsampling_factor)
    image = image.permute(0, 2, 3, 4, 1)
    print(f"second permutation shape {image.shape}")
    return image


def save_images(images):
    for i, image in enumerate(images):
        image = image.squeeze().cpu().numpy()
        tiff.imwrite(f"image_{i}.tif", image)


if __name__ == "__main__":
    images = load_images("./training/source/", 192)
    print(images.shape)
    # save_images(images)

    images = CustomDataset(images, 2)
    # images = images.astype(np.float32)
    # images = downsample_permute(images, 2)
    print("pre-upscaling")
    im_new = []
    for im in images:
        im = np.expand_dims(im, axis=0)
        print(im.shape)
        im = upsample_permute(im, 2)
        im_new.append(im)
    save_images(im_new)
