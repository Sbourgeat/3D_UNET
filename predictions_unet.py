import os
import torch
import numpy as np
import tifffile as tiff
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset
from sweep_UNET import UNet3D
from tqdm import tqdm


# Configuration parameters
class Config:
    model_path = "./model_3d_opti.pth"  # Change to your model path
    input_dir = "./training/source/"  # Change to your input images directory
    output_dir = "./predictions"  # Change to your output directory
    batch_size = 1
    target_shape = 192
    downsampling_factor = 6  # Use the same downsampling factor as during training


def load_images(directory_path, target_shape):
    images = []
    target_shape = (target_shape, target_shape, target_shape)
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".tif"):
            image = tiff.imread(os.path.join(directory_path, filename))
            if image.shape != target_shape:
                image = resize(
                    image, target_shape, preserve_range=True, anti_aliasing=True
                )
                image = image.astype(np.float32)
            images.append(image)
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)  # Add channel dimension
    return images


def normalize(source_images):
    normalized_images = []
    for img in source_images:
        min_val = np.min(img)
        max_val = np.max(img)
        normalized_img = (img - min_val) / (max_val - min_val + np.finfo(float).eps)
        normalized_images.append(normalized_img)
    normalized_images = np.array(normalized_images)
    normalized_images = np.clip(normalized_images, 0, 1)
    normalized_images = normalized_images.astype(np.float32)
    return normalized_images


class CustomDataset(Dataset):
    def __init__(self, source_images, downsampling_factor):
        self.source_images = torch.tensor(source_images, dtype=torch.float32)
        self.source_images = self.source_images.permute(0, 4, 2, 3, 1)
        self.source_images = torch.nn.functional.interpolate(
            self.source_images, scale_factor=1 / downsampling_factor
        )
        self.source_images = self.source_images.permute(0, 2, 3, 4, 1)

    def __len__(self):
        return len(self.source_images)

    def __getitem__(self, idx):
        return self.source_images[idx]


def save_predictions(predictions, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, prediction in enumerate(predictions):
        prediction = prediction  # Remove the channel dimension
        tiff.imwrite(
            os.path.join(output_dir, f"prediction_{idx}.tif"),
            prediction.astype(np.float32),
        )


def mask_and_save_images(input_images, predictions, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (input_image, prediction) in enumerate(zip(input_images, predictions)):
        input_image = input_image  # Remove the channel dimension
        mask = (prediction > 0.5).astype(np.float32)
        masked_image = input_image * mask
        tiff.imwrite(
            os.path.join(output_dir, f"masked_image_{idx}.tif"),
            masked_image.astype(np.float32),
        )


def upsample_and_save_images(input_images, predictions, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (input_image, prediction) in enumerate(zip(input_images, predictions)):
        input_image = input_image  # Remove the channel dimension
        upsampled_image = resize(
            prediction,
            (config.target_shape, config.target_shape, config.target_shape, 1),
            preserve_range=True,
            anti_aliasing=True,
        )
        tiff.imwrite(
            os.path.join(output_dir, f"upsampled_image_{idx}.tif"),
            upsampled_image.astype(np.float32),
        )


if __name__ == "__main__":
    config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = UNet3D(dropout=0)  # Set dropout to 0 for inference
    model.load_state_dict(torch.load(config.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load the input images
    input_images = load_images(config.input_dir, config.target_shape)
    original_shape = input_images[0].shape
    input_images = normalize(input_images)

    # Create the dataset and dataloader
    dataset = CustomDataset(
        input_images, downsampling_factor=config.downsampling_factor
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Perform predictions
    predictions = []
    with torch.no_grad():
        for inputs in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    # Save the predictions
    # save_predictions(predictions, config.output_dir)
    # print(f"Predictions saved to {config.output_dir}.")

    # Mask the images and save the masked images
    mask_and_save_images(dataset.source_images.numpy(), predictions, config.output_dir)
    print(f"Masked images saved to {config.output_dir}.")

    # Upsample the images and save the upsampled images
    # upsample_and_save_images(
    #    dataset.source_images.numpy(), predictions, config.output_dir
    # )
    # print(f"Upsampled images saved to {config.output_dir}.")
