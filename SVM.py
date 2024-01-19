import torch
import DataLoad as DL
import numpy as np
from sklearn import svm

root_path = 'Data'
device = 'cuda' if torch. cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)

img_size = 48*48*3
num_classes = 2

train_dataset, validation_dataset, test_dataset = DL.split_datasets(DL.create_datasets(root_path))
train_dataset, validation_dataset, test_dataset = DL.augmentation_datasets(train_dataset, validation_dataset, test_dataset)
train_loader, val_loader, test_loader = DL.data_load(train_dataset, validation_dataset, test_dataset, batch_size=0)

train_data, train_label = DL.SVM_data_setting(train_loader)
val_data, val_label = DL.SVM_data_setting(val_loader)
test_data, test_label = DL.SVM_data_setting(test_loader)

train_data = np.append(train_data, val_data, axis=0)
train_label = np.append(train_label, val_label)

print(len(train_dataset), len(validation_dataset), len(test_dataset))
print(len(train_data), len(test_data))

model = svm.SVC(kernel='rbf', gamma='auto')

import time
import datetime
print('Training...')
start = time.time()
model.fit(train_data, train_label)
final = time.time()-start
print('Training Finish')
score = model.score(test_data, test_label)*100
result_list = str(datetime.timedelta(seconds=final)).split(".")

print(f'Test Accuracy : {score:.2f}%, Time : {result_list[0]}')