import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os
import platform

class MNIST_DNN(nn.Module):
    def __init__(self):
        super(MNIST_DNN, self).__init__()
        # First conv layer
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second conv layer
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # 14x14 -> 14x14
        self.bn2 = nn.BatchNorm2d(16)
        
        # Third conv layer
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)  # 7x7 -> 7x7
        self.bn3 = nn.BatchNorm2d(32)
        
        # Fourth conv layer
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)  # 3x3 -> 3x3
        self.bn4 = nn.BatchNorm2d(32)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 3 * 3, 32)  # 288 -> 32
        self.fc2 = nn.Linear(32, 10)  # 32 -> 10
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # First block: 28x28 -> 14x14
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        
        # Second block: 14x14 -> 7x7
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7
        
        # Third block: 7x7 -> 3x3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 7x7 -> 3x3
        
        # Fourth block: 3x3 -> 3x3
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten and FC layers
        x = x.view(-1, 32 * 3 * 3)  # 3x3x32 = 288
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, accuracy):
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Get system info
    device = "cpu"  # Since we're using CPU
    system = platform.system()
    
    # Create filename with details
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accuracy_str = f"{accuracy:.2f}".replace(".", "p")
    model_path = f"models/mnist_model_{accuracy_str}acc_{device}_{system}_{timestamp}.pth"
    
    # Save model
    torch.save(model.state_dict(), model_path)
    return model_path 