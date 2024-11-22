import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 2 * 2, 512)  # Adjusted for 20x20 input
        self.fc2 = nn.Linear(8, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 10)
        
    def forward(self, x):
        f = x[:,400:]
        x = x[:,:400].view(-1, 1, 20, 20)
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(-1, 128 * 2 * 2)
        x = nn.ReLU()(self.fc1(x))
        f = nn.ReLU()(self.fc2(f))
        f = nn.ReLU()(self.fc3(f))
        x = torch.cat((x, f), 1)
        x = nn.ReLU()(self.fc4(x))
        x = self.fc5(x)
        return x
    
# Initialize the model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.01)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)