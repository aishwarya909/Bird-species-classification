from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os
import kagglehub
import torchvision.transforms as transforms

# Download dataset from KaggleHub
path = kagglehub.dataset_download("gpiosenka/birdies")
print("Path to dataset files:", path)

# Hyperparameters
batch_size = 32
input_size = 224

# Transforms
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset directories
train_dir = os.path.join(path, "train")
test_dir = os.path.join(path, "test")

# Datasets
train_dataset = ImageFolder(train_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)

# Split train into train/val
train_len = int(0.8 * len(train_dataset))
val_len = len(train_dataset) - train_len
train_data, val_data = random_split(train_dataset, [train_len, val_len])

# Dataloaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
