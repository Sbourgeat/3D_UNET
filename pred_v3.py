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


def load_and_preprocess_images(directory_path, config):
    images = []
    filenames = []
    original_shapes = []
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".tif"):
            image = tiff.imread(os.path.join(directory_path, filename))
            original_shapes.append(image.shape)
            # Resize and then normalize image
            image = resize(
                image,
                (config.target_shape, config.target_shape, config.target_shape),
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)
            image = normalize(image)
            images.append(image)
            filenames.append(filename)
    # Add channel dimension and adjust tensor dimensions for PyTorch
    images = np.array(images).reshape(
        -1, config.target_shape, config.target_shape, config.target_shape, 1
    )
    images = torch.tensor(images).permute(0, 4, 1, 2, 3)  # N, C, D, H, W
    print(f"input image shape is {images.shape}")
    return images, filenames, original_shapes


def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val + np.finfo(float).eps)


class CustomDataset(Dataset):
    def __init__(self, source_images):
        self.source_images = source_images

    def __len__(self):
        return len(self.source_images)

    def __getitem__(self, idx):
        return self.source_images[idx]


def upsample_and_save_predictions(predictions, filenames, original_shapes, config):
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    for prediction, filename, original_shape in zip(
        predictions, filenames, original_shapes
    ):
        prediction_tensor = torch.tensor(prediction, dtype=torch.float32).unsqueeze(0)
        # Reshape to original dimensions
        upsampled_prediction = (
            torch.nn.functional.interpolate(
                prediction_tensor,
                size=original_shape,
                mode="trilinear",
                align_corners=True,
            )
            .squeeze(0)
            .detach()
            .numpy()
        )

        print(
            f"Original shape: {original_shape}, Upsampled prediction shape: {upsampled_prediction.shape}"
        )
        tiff.imwrite(
            os.path.join(config.output_dir, f"upsampled_{filename}"),
            upsampled_prediction.astype(np.float32),
        )


if __name__ == "__main__":
    config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = UNet3D(dropout=0)  # Set dropout to 0 for inference
    model.load_state_dict(torch.load(config.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load and preprocess the input images
    input_images, filenames, original_shapes = load_and_preprocess_images(
        config.input_dir, config
    )

    # Create the dataset and dataloader
    dataset = CustomDataset(input_images)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Perform predictions
    predictions = []
    with torch.no_grad():
        for inputs in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs).permute(0, 4, 1, 2, 3)
            print(f"Output shape is {outputs.shape}")
            predictions.extend(outputs.cpu().numpy())

    # Upsample the predictions and save them
    upsample_and_save_predictions(predictions, filenames, original_shapes, config)
    print(f"Upscaled predictions saved to {config.output_dir}.")
