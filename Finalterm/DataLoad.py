from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torch
import numpy as np

# from PIL import Image
# import os

# def png_RGBA():
#     folder_path_list = ["Data/mask", "Data/no_mask"]
#     for folder_path in folder_path_list:
#         png_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
#         for png_file in png_files:
#             image = Image.open(os.path.join(folder_path, png_file))
#             if image.mode != "RGBA":
#                 image = image.convert("RGBA")
#             image.save(os.path.join(folder_path, png_file))

class TrainDataset(Dataset):
    def __init__(self, train, transform=None):
        self.train = train
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.train[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.train)

class TestDataset(Dataset):
    def __init__(self, test, transform=None):
        self.test = test
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.test[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.test)
    
def create_datasets(file_path):
    return datasets.ImageFolder(root=file_path, transform=None)

def split_datasets(dataset):

    train_ratio, val_ratio = [0.7, 0.15]

    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, validation_dataset, test_dataset

def augmentation_datasets(train_data, val_data, test_data):

    train_transform=transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=2.0),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    test_transform=transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    return TrainDataset(train_data, train_transform), TestDataset(val_data, test_transform), TestDataset(test_data, test_transform)

def data_load(train_dataset, validation_dataset, test_dataset, batch_size=0):
    
    if batch_size == 0:
        train_loader = DataLoader(train_dataset, 
                            shuffle=True, 
                            num_workers=0
                            )

        val_loader = DataLoader(validation_dataset, 
                            shuffle=True, 
                            num_workers=0
                            )

        test_loader = DataLoader(test_dataset, 
                            shuffle=False, 
                            num_workers=0
                            )
    else:
        train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size,
                            shuffle=True, 
                            num_workers=0
                            )

        val_loader = DataLoader(validation_dataset, 
                            batch_size=batch_size,
                            shuffle=True, 
                            num_workers=0
                            )

        test_loader = DataLoader(test_dataset, 
                            batch_size=batch_size,
                            shuffle=False, 
                            num_workers=0
                            )
        
    return train_loader, val_loader, test_loader

###########################
# RNN
###########################
class StockDataset(Dataset):
    def __init__(self, data, window):
        self.data = data
        self.window = window

    def __len__(self):
        return len(self.data) - self.window

    def __getitem__(self, idx):
        past = self.data[idx:idx+self.window]
        future = self.data[idx+self.window]
        return torch.tensor(past, dtype=torch.float32), torch.tensor(future, dtype=torch.float32)
    
def loader(dataset, batch_size=0):
    
    if batch_size == 0:
        data_loader = DataLoader(dataset, 
                            shuffle=False, 
                            num_workers=0
                            )
    else:
        data_loader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=False, 
                            num_workers=0
                            )
        
    return data_loader

