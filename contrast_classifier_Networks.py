import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNet18SingleChannel(nn.Module):
    # Define the ResNet18 model with a single image channel and an output value between 0 and 1
    def __init__(self, num_classes=2):
        super(ResNet18SingleChannel, self).__init__()
        # Load the pre-trained ResNet18 model
        resnet = models.resnet18(pretrained=False)
        # Modify the first convolutional layer to take a single channel input
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        # Modify the final fully connected layer to output a single value

        self.resnet = resnet

        #final fc to go from [batch_size, 1000] to [batch_size, num_classes]
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
    

