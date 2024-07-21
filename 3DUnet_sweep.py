import tifffile as tiff
import torch
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim

# from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import torchio as tio

from skimage.transform import resize
import os
from tqdm import tqdm

# import matplotlib.pyplot as plt
import numpy as np
from sweep_UNET import UNet3D
import wandb
from ranger import Ranger
from adabelief_pytorch import AdaBelief

print("PyTorch version: {}".format(torch.__version__))
print("Tifffile version: {}".format(tiff.__version__))
print("Numpy version: {}".format(np.__version__))

print("Dependencies installed and imported.")

### CHECK THE USE OF GPU ###########################################################################
print("CUDA available: ", torch.cuda.is_available())
print("Num of available GPUs: ", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Current GPU: ", torch.cuda.get_device_name(torch.cuda.current_device()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


wandb.init(project="pytorch_3dUnet")

config = wandb.config


### VARIABLES ######################################################################################


class CustomDataset(Dataset):
    def __init__(
        self, source_images, target_images, downsampling_factor, augment=False
    ):
        self.source_images = torch.tensor(source_images, dtype=torch.float32)
        self.source_images = self.source_images.permute(0, 4, 1, 2, 3)
        self.source_images = torch.nn.functional.interpolate(
            self.source_images, scale_factor=1 / downsampling_factor
        )
        self.source_images = self.source_images.permute(0, 2, 3, 4, 1)
        self.target_images = torch.tensor(target_images, dtype=torch.float32)
        self.target_images = self.target_images.permute(0, 4, 1, 2, 3)
        self.target_images = torch.nn.functional.interpolate(
            self.target_images, scale_factor=1 / downsampling_factor
        )
        self.target_images = self.target_images.permute(0, 2, 3, 4, 1)
        self.augment = augment

        # Define the augmentation pipeline
        # self.transforms = tio.Compose(
        #    [
        #        tio.RandomFlip(axes=(0, 1, 2), p=0.5),
        #        tio.RandomAffine(
        #           scales=(0.9, 1.1),
        #        degrees=10,
        #        isotropic=False,
        #        center="image",
        #        p=0.75,
        #    ),
        #       tio.RandomElasticDeformation(
        #       num_control_points=(7, 7, 7), max_displacement=(5, 5, 5), p=0.75
        #   ),
        #       tio.RandomNoise(mean=0, std=0.1, p=0.25),
        #       tio.RandomBlur(p=0.25),
        #   ]
        # )

    def __len__(self):
        return len(self.source_images)

    def shape(self):
        return self.source_images.shape

    def __getitem__(self, idx):
        source, target = self.source_images[idx], self.target_images[idx]
        if self.augment:
            sample = tio.Subject(
                source=tio.ScalarImage(tensor=source),
                target=tio.LabelMap(tensor=target),
            )
            sample = self.transforms(sample)
            source, target = sample.source.tensor, sample.target.tensor
        return source, target


# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


# Combined BCE and Dice Loss
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# Tversky Loss
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        true_pos = (inputs * targets).sum()
        false_neg = ((1 - inputs) * targets).sum()
        false_pos = (inputs * (1 - targets)).sum()
        tversky = (true_pos + smooth) / (
            true_pos + self.alpha * false_neg + self.beta * false_pos + smooth
        )
        return 1 - tversky


### LOAD IMAGES ####################################################################################
def load_images(directory_path, target_shape):  # .tif files to numpy arrays
    images = []
    target_shape = (target_shape, target_shape, target_shape)
    # print(f'target_shape = {target_shape}')
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".tif"):
            image = tiff.imread(os.path.join(directory_path, filename))
            if image.shape != target_shape:
                image = resize(
                    image, target_shape, preserve_range=True, anti_aliasing=True
                )
                image.astype(np.float32)
            images.append(image)
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)  # Add channel dimension
    return images


train_source_dir = "/home/sbourgeat/Project/MachineLearning/UNET/training/source/"  # "/home/sbourgeat/Project/MachineLearning/UNET/cropped_training/cropped_training/source"
train_target_dir = "/home/sbourgeat/Project/MachineLearning/UNET/training/target/"  # "/home/sbourgeat/Project/MachineLearning/UNET/cropped_training/cropped_training/target_binarized"
test_source_dir = "/home/sbourgeat/Project/MachineLearning/UNET/test/source/"  # #(
#    "/home/sbourgeat/Project/MachineLearning/UNET/cropped_test/cropped_test/source"
# )
test_target_dir = "/home/sbourgeat/Project/MachineLearning/UNET/test/target/"  # #"/home/sbourgeat/Project/MachineLearning/UNET/cropped_test/cropped_test/target_binarized"
# Load datasets
train_source = load_images(train_source_dir, config.target_shape)
train_target = load_images(train_target_dir, config.target_shape)
test_source = load_images(test_source_dir, config.target_shape)
test_target = load_images(test_target_dir, config.target_shape)


### NORMALIZE ######################################################################################


def normalize(source_images):  # Numpy array to numpy array
    normalized_images = []
    for img in source_images:
        min_val = np.min(img)
        max_val = np.max(img)
        normalized_img = (img - min_val) / (max_val - min_val + np.finfo(float).eps)
        normalized_images.append(normalized_img)
        # print("Image mean from: ", np.mean(img), " to: ", np.mean(normalized_img))

    normalized_images = np.array(normalized_images)
    normalized_images = np.clip(normalized_images, 0, 1)
    normalized_images = normalized_images.astype(np.float32)
    return normalized_images


def binarize_targets(target_path, threshold=0.1):
    binarized_image = []
    for targets in target_path:
        targets[targets >= threshold] = 1
        targets[targets < threshold] = 0
        binarized_image.append(targets)

    targets = np.array(binarized_image)
    targets = np.clip(targets, 0, 1)
    targets = targets.astype(np.float32)
    return targets


train_source = normalize(train_source)
test_source = normalize(test_source)

train_target = binarize_targets(train_target)
test_target = binarize_targets(test_target)

train_dataset = CustomDataset(
    train_source, train_target, config.downsampling_factor, augment=config.augment
)
test_dataset = CustomDataset(
    test_source, test_target, config.downsampling_factor, augment=False
)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

print("length of training data: ", len(train_source))
print("length of test data: ", len(test_source))
print("shape of training data: ", train_source.shape, train_target.shape)
print("shape of test data: ", test_source.shape, test_target.shape)


### MODEL CREATION #################################################################################
model = UNet3D(dropout=config.dropout)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Select the loss function
if config.loss_function == "bce":
    criterion = nn.BCELoss()
elif config.loss_function == "dice":
    criterion = DiceLoss()
elif config.loss_function == "bce_dice":
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
elif config.loss_function == "tversky":
    criterion = TverskyLoss(alpha=0.7, beta=0.3)

# Select the optimizer
if config.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
elif config.optimizer == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
elif config.optimizer == "ranger":
    optimizer = Ranger(model.parameters(), lr=config.learning_rate)
elif config.optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
elif config.optimizer == "nag":
    optimizer = optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=0.9, nesterov=True
    )
elif config.optimizer == "adabelief":
    optimizer = AdaBelief(model.parameters(), lr=config.learning_rate)

# Select the learning rate scheduler
if config.lr_scheduler == "step":
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
elif config.lr_scheduler == "plateau":
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )
elif config.lr_scheduler == "cosine":
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


# Early stopping parameters
patience = 20
best_val_loss = float("inf")
early_stop_counter = 0


def dice_coefficient(preds, targets, smooth=1):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()


# Training and evaluation loop
def evaluate(model, dataloader, criterion, device, epoch):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    dice_score = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.numel()
            dice_score += dice_coefficient(predicted, targets) * inputs.size(0)
    val_loss /= len(dataloader.dataset)
    accuracy = correct / total
    dice_score /= len(dataloader.dataset)
    wandb.log(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "val_accuracy": accuracy,
            "val_dice": dice_score,
        }
    )
    return val_loss, accuracy, dice_score


for epoch in range(config.num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    with tqdm(
        total=len(train_loader),
        desc=f"Epoch {epoch + 1}/{config.num_epochs}",
        unit="batch",
    ) as pbar:
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.numel()
            pbar.update(1)
            wandb.log({"train_loss": loss.item()})
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct / total
    wandb.log(
        {"epoch": epoch, "train_loss": epoch_loss, "train_accuracy": epoch_accuracy}
    )
    val_loss, val_accuracy, val_dice = evaluate(
        model, test_loader, criterion, device, epoch
    )

    if config.lr_scheduler in ["step", "cosine"]:
        scheduler.step()
    elif config.lr_scheduler == "plateau":
        scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_model_3d_aug.pth")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

torch.save(model.state_dict(), "model_3d_aug.pth")
wandb.finish()
