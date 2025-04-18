import torchvision.models as models
import torch.nn as nn

def get_model(model_name, num_classes, dropout_rate=0.5):
    if model_name == "vgg":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(4096, num_classes)
        )
    elif model_name == "resnet":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
    return model
