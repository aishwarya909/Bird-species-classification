from src.download_dataset import download_bird_dataset
from src.dataloader import get_dataloaders
from src.models import get_model
from src.train import train_model
from src.evaluate import evaluate

# Hyperparameters
input_size = 224
batch_size = 32
learning_rate = 0.001
dropout_rate = 0.5
num_epochs = 10
optimizer_type = "adam"

# Step 1: Download dataset
path = download_bird_dataset()

# Step 2: Load data
train_loader, val_loader, test_loader, train_dataset = get_dataloaders(path, input_size, batch_size)
num_classes = len(train_dataset.dataset.classes)

# Step 3: Train VGG model
vgg_model = get_model("vgg", num_classes, dropout_rate)
vgg_model = train_model(vgg_model, train_loader, val_loader, "vgg", num_epochs, learning_rate, optimizer_type)
evaluate(vgg_model, test_loader)

# Step 4: Train ResNet model
resnet_model = get_model("resnet", num_classes, dropout_rate)
resnet_model = train_model(resnet_model, train_loader, val_loader, "resnet", num_epochs, learning_rate, optimizer_type)
evaluate(resnet_model, test_loader)
