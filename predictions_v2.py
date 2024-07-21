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
    target_shape = 64
    downsampling_factor = 2  # Use the same downsampling factor as during training


def load_images(directory_path):
    images = []
    filenames = []
    original_shapes = []
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".tif"):
            image = tiff.imread(os.path.join(directory_path, filename))
            original_shapes.append(image.shape)
            image = resize(
                image,
                (config.target_shape, config.target_shape, config.target_shape),
                preserve_range=True,
                anti_aliasing=True,
            )
            image = image.astype(np.float32)
            images.append(image)
            filenames.append(filename)
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)  # Add channel dimension
    return images, filenames, original_shapes


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
        print(f"before permutation shape {self.source_images.shape}")
        self.source_images = self.source_images.permute(0, 4, 1, 2, 3)
        print(f"after permutation shape {self.source_images.shape}")
        self.source_images = torch.nn.functional.interpolate(
            self.source_images, scale_factor=1 / downsampling_factor
        )
        self.source_images = self.source_images.permute(0, 2, 3, 4, 1)
        print(f"after interpolation shape {self.source_images.shape}")

    def __len__(self):
        return len(self.source_images)

    def __getitem__(self, idx):
        return self.source_images[idx]


def upsample_and_save_predictions(predictions, filenames, original_shapes, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for _, (prediction, filename, original_shape) in enumerate(
        zip(predictions, filenames, original_shapes)
    ):
        # prediction = prediction.squeeze()  # Remove the channel dimensio
        print(f" shape {prediction.shape} ")
        prediction = torch.tensor(prediction, dtype=torch.float32)
        prediction = torch.nn.functional.interpolate(
            prediction, size=original_shape, mode="trilinear", align_corners=True
        )
        # to numpy
        prediction = prediction.detach().cpu().numpy()

        print(
            f"Original shape: {original_shape}, Upsampled prediction shape: {prediction.shape}"
        )
        tiff.imwrite(
            os.path.join(output_dir, f"upsampled_{filename}"),
            prediction.astype(np.float32),
        )


def apply_mask(original, prediction):
    # Resize prediction to match the original image dimensions
    # Ensure prediction is binary (0 or 1) if it's not already
    binary_mask = (prediction > 0.5).astype(np.float32)
    # Apply the mask to the original image
    masked_image = original * binary_mask
    return masked_image


if __name__ == "__main__":
    config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = UNet3D(dropout=0)  # Set dropout to 0 for inference
    model.load_state_dict(torch.load(config.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load the input images
    input_images, filenames, original_shapes = load_images(config.input_dir)
    input_images = normalize(input_images)

    # Create the dataset and dataloader
    dataset = CustomDataset(
        input_images, downsampling_factor=config.downsampling_factor
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Perform predictions
    predictions = []
    names = []
    with torch.no_grad():
        for inputs, o_shape, name in tqdm(zip(dataloader, original_shapes, filenames)):
            inputs = inputs.to(device)
            outputs = model(inputs).permute(0, 4, 1, 2, 3)
            print(f"output shape is {outputs.shape}")
            outputs = torch.nn.functional.interpolate(
                outputs, size=o_shape, mode="trilinear", align_corners=True
            )
            outputs = outputs.cpu().numpy()
            # tiff.imwrite(
            #    os.path.join(config.output_dir, f"prediction_{name}"),
            #    outputs.astype(np.float32),
            # )
            names.append(name)
            predictions.append(outputs)
    # Upsample the predictions and save them

    # mask images

    for i in range(len(predictions)):
        print(f"Masking image {names[i]}")
        image = tiff.imread(os.path.join(config.input_dir, names[i]))
        mask = apply_mask(image, predictions[i])
        tiff.imwrite(
            os.path.join(config.output_dir, f"masked_image_{names[i]}"),
            mask.astype(np.float32),
        )

    print(f"Upscaled predictions saved to {config.output_dir}.")
