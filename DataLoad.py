from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.transforms as transforms

root_path = 'Data'

'''Data Load'''

dataset = datasets.ImageFolder(root=root_path, 
                                    transform=transforms.Compose([
                                        transforms.Resize((256, 256)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomCrop(224),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])
                              )

classes = dataset.classes

print(dataset.classes)
print(dataset.class_to_idx)
print("\n")

'''Data split'''

dataset_size = len(dataset)
train_size = int(dataset_size * 0.7)
validation_size = int(dataset_size * 0.15)
test_size = dataset_size - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

print(f"Training Data Size : {len(train_dataset)}")
print(f"Validation Data Size : {len(validation_dataset)}")
print(f"Testing Data Size : {len(test_dataset)}")

batch_size = 32 # batch_size 지정

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0
                         )

test_loader = DataLoader(test_dataset, 
                         batch_size=batch_size,
                         shuffle=False, 
                         num_workers=0
                        )