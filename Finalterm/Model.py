import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes, channel1, channel2, kernel, output_layer, drop):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, channel1, kernel_size=kernel, padding=kernel//2)
        self.conv2 = nn.Conv2d(channel1, channel2, kernel_size=kernel, padding=kernel//2)
        self.conv3_drop = nn.Dropout2d(p = drop)
        self.fc1 = nn.Linear(channel2*12*12, output_layer)
        self.fc2 = nn.Linear(output_layer, num_classes)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv2(x)), 2))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        self.rnn.flatten_parameters()
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out