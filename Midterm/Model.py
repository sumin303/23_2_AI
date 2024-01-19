import torch
import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self, input_size, num_classes, input_layer, hidden_layer, drop):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, input_layer)
        self.fc2 = nn.Linear(input_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, num_classes)
        self.dropout = nn.Dropout(p=drop)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return x