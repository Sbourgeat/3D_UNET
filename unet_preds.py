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
    pass


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


def save_predictions(predictions, output_dir, original_shape):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, prediction in enumerate(predictions):
        prediction = prediction.squeeze()  # Remove the channel dimension
        # prediction = resize(
        #    prediction, original_shape, preserve_range=True, anti_aliasing=True
        # )
        tiff.imwrite(
            os.path.join(output_dir, f"prediction_{idx}.tif"),
            prediction.astype(np.float32),
        )


if __name__ == "__main__":
    # Hard-coded configuration
    config = Config()
    config.model_path = "./model_3d_opti.pth"  # Change to your model path
    config.input_dir = "./training/source/"  # Change to your input images directory
    config.output_dir = "./predictions"  # Change to your output directory
    config.batch_size = 1
    config.target_shape = 192
    config.downsampling_factor = (
        6  # Use the same downsampling factor as during training
    )

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
            # outputs = torch.nn.functional.interpolate(
            #    outputs,
            #    scale_factor=config.downsampling_factor,
            #    mode="trilinear",
            #    align_corners=False,
            # )  # Upscale to original size
            predictions.extend(outputs.cpu().numpy())

    # Save the predictions
    save_predictions(predictions, config.output_dir, original_shape)
    print(f"Predictions saved to {config.output_dir}.")
